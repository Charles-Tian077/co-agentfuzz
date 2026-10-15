from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

from co_agent_fuzz.data import read_jsonl


def template_fingerprint(record: Dict) -> str:
    text = f"{record.get('pre_prompt', '')}\n{record.get('post_prompt', '')}".lower()
    text = re.sub(r"['\"]?[a-z0-9_-]{4,}['\"]?", "<TOKEN>", text)
    text = " ".join(text.split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def defense_only(record: Dict) -> Dict:
    return {key: record.get(key) for key in
            ("sample_id", "pre_prompt", "post_prompt", "access_code")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-resistant Co-AgentFuzz splits")
    parser.add_argument("--mode", choices=["hijacking", "extraction"], required=True)
    parser.add_argument("--source")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.validation_ratio <= 0 or args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("Ratios must be positive and train + validation must be below 1")
    source = Path(args.source) if args.source else dataset_root() / f"{args.mode}_robustness_dataset.jsonl"
    output = Path(args.output_dir) if args.output_dir else dataset_root() / "co_agent_fuzz" / args.mode
    output.mkdir(parents=True, exist_ok=True)
    expected = [output / name for name in (
        "pre_jailbreak_seeds.jsonl", "train_defenses.jsonl",
        "validation_defenses.jsonl", "test_defenses.jsonl", "split_manifest.json")]
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing split files: {existing}")

    records = read_jsonl(source)
    groups: Dict[str, List[Dict]] = {}
    for record in records:
        groups.setdefault(template_fingerprint(record), []).append(record)
    keys = list(groups)
    random.Random(args.seed).shuffle(keys)
    train_end = round(len(keys) * args.train_ratio)
    validation_end = train_end + round(len(keys) * args.validation_ratio)
    partitions = {
        "train": keys[:train_end],
        "validation": keys[train_end:validation_end],
        "test": keys[validation_end:],
    }
    split_records = {
        name: [record for key in selected for record in groups[key]]
        for name, selected in partitions.items()
    }
    seed_records = []
    seen_attacks = set()
    for record in split_records["train"]:
        attack = str(record.get("attack", "")).strip()
        normalised_attack = " ".join(attack.lower().split())
        if not attack or normalised_attack in seen_attacks:
            continue
        seen_attacks.add(normalised_attack)
        seed_records.append({
            "sample_id": record.get("sample_id"), "attack": attack, "split": "train"})
    write_jsonl(output / "pre_jailbreak_seeds.jsonl", seed_records)
    write_jsonl(output / "train_defenses.jsonl", map(defense_only, split_records["train"]))
    write_jsonl(output / "validation_defenses.jsonl", map(defense_only, split_records["validation"]))
    write_jsonl(output / "test_defenses.jsonl", map(defense_only, split_records["test"]))
    manifest = {
        "mode": args.mode,
        "source": (str(source.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
                   if source.resolve().is_relative_to(REPO_ROOT.resolve())
                   else str(source.resolve())),
        "random_seed": args.seed,
        "grouping": "normalised pre_prompt + post_prompt template fingerprint",
        "group_counts": {name: len(selected) for name, selected in partitions.items()},
        "record_counts": {name: len(selected) for name, selected in split_records.items()},
        "unique_train_seeds": len(seed_records),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
