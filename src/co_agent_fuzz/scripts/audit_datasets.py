from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

from co_agent_fuzz.data import read_jsonl


def normalise(text: object) -> str:
    return " ".join(str(text or "").lower().split())


def digest(text: object) -> str:
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


def quantile(values: List[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyse_file(path: Path) -> Dict:
    records = read_jsonl(path)
    ids = [str(item.get("sample_id", "")) for item in records]
    attacks = [normalise(item.get("attack")) for item in records]
    lengths = [len(item) for item in attacks]
    missing = Counter()
    for record in records:
        for field in ("sample_id", "attack", "pre_prompt", "post_prompt", "access_code"):
            if field not in record or record[field] in (None, ""):
                missing[field] += 1
    return {
        "file": path.name,
        "count": len(records),
        "unique_sample_ids": len(set(ids)),
        "duplicate_sample_ids": len(ids) - len(set(ids)),
        "unique_attacks": len(set(attacks)),
        "duplicate_attacks": len(attacks) - len(set(attacks)),
        "missing_fields": dict(missing),
        "attack_length": {
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "p50": quantile(lengths, 0.50),
            "p95": quantile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "non_ascii_attacks": sum(any(ord(char) > 127 for char in text) for text in attacks),
        "control_character_attacks": sum(any(not char.isprintable() for char in text) for text in attacks),
        "_id_set": set(ids),
        "_attack_set": {digest(item) for item in attacks},
        "_defense_set": {digest(f"{item.get('pre_prompt', '')}\n{item.get('post_prompt', '')}")
                         for item in records},
    }


def serialisable(report: Dict) -> Dict:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PromptFuzz/Co-AgentFuzz JSONL datasets")
    parser.add_argument("--dataset-dir", default=str(dataset_root()))
    parser.add_argument("--output", default=str(output_root() / "dataset_audit.json"))
    args = parser.parse_args()
    paths = sorted(Path(args.dataset_dir).glob("*.jsonl"))
    reports = [analyse_file(path) for path in paths]
    overlaps = []
    for left, right in combinations(reports, 2):
        left_mode = left["file"].split("_", 1)[0]
        right_mode = right["file"].split("_", 1)[0]
        if left_mode != right_mode:
            continue
        id_overlap = len(left["_id_set"] & right["_id_set"])
        attack_overlap = len(left["_attack_set"] & right["_attack_set"])
        defense_overlap = len(left["_defense_set"] & right["_defense_set"])
        if id_overlap or attack_overlap or defense_overlap:
            overlaps.append({
                "left": left["file"], "right": right["file"],
                "sample_id_overlap": id_overlap,
                "attack_overlap": attack_overlap,
                "defense_template_overlap": defense_overlap,
            })
    output = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "files": [serialisable(item) for item in reports],
        "cross_file_overlaps": overlaps,
        "recommendation": (
            "Use group-disjoint train/validation/test splits. Mutator/RL training may use train; "
            "threshold selection may use validation; report final ASR only on untouched test defenses."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audited {len(reports)} files -> {destination.resolve()}")
    for report in reports:
        print(f"  {report['file']}: n={report['count']} duplicate_ids={report['duplicate_sample_ids']} "
              f"duplicate_attacks={report['duplicate_attacks']} p95_len={report['attack_length']['p95']:.0f}")
    print(f"Cross-file overlap pairs requiring review: {len(overlaps)}")
    for item in sorted(overlaps, key=lambda row: row["sample_id_overlap"] + row["attack_overlap"], reverse=True)[:15]:
        print(f"  {item['left']} <-> {item['right']}: ids={item['sample_id_overlap']} "
              f"attacks={item['attack_overlap']} defenses={item['defense_template_overlap']}")


if __name__ == "__main__":
    main()
