from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

from co_agent_fuzz.data import read_jsonl
from co_agent_fuzz.scripts.build_splits import template_fingerprint


def normalise(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_mode(mode_dir: Path) -> dict:
    partition_names = ("train", "validation", "test")
    paths = {name: mode_dir / f"{name}_defenses.jsonl" for name in partition_names}
    seed_path = mode_dir / "pre_jailbreak_seeds.jsonl"
    missing_files = [str(path) for path in (*paths.values(), seed_path) if not path.exists()]
    if missing_files:
        return {"mode": mode_dir.name, "passed": False, "violations": [
            {"check": "required_files", "detail": path} for path in missing_files
        ]}

    records = {name: read_jsonl(path) for name, path in paths.items()}
    seeds = read_jsonl(seed_path)
    violations = []
    ids = {}
    fingerprints = {}

    for name, items in records.items():
        item_ids = [str(item.get("sample_id", "")) for item in items]
        ids[name] = set(item_ids)
        fingerprints[name] = {template_fingerprint(item) for item in items}
        if len(item_ids) != len(ids[name]):
            violations.append({"check": "unique_sample_id", "partition": name,
                               "count": len(item_ids) - len(ids[name])})
        leaked_attacks = sum(bool(normalise(item.get("attack"))) for item in items)
        if leaked_attacks:
            violations.append({"check": "defense_has_attack", "partition": name,
                               "count": leaked_attacks})
        missing_required = sum(
            not str(item.get("sample_id", "")).strip()
            or not (str(item.get("pre_prompt", "")).strip()
                    or str(item.get("post_prompt", "")).strip())
            or not str(item.get("access_code", "")).strip()
            for item in items
        )
        if missing_required:
            violations.append({"check": "required_fields", "partition": name,
                               "count": missing_required})

    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        id_overlap = ids[left] & ids[right]
        template_overlap = fingerprints[left] & fingerprints[right]
        if id_overlap:
            violations.append({"check": "cross_split_sample_id", "partitions": [left, right],
                               "count": len(id_overlap)})
        if template_overlap:
            violations.append({"check": "cross_split_template", "partitions": [left, right],
                               "count": len(template_overlap)})

    seed_attacks = [normalise(item.get("attack")) for item in seeds]
    empty_seeds = sum(not attack for attack in seed_attacks)
    duplicate_seeds = len(seed_attacks) - len(set(seed_attacks))
    if empty_seeds:
        violations.append({"check": "nonempty_seed", "count": empty_seeds})
    if duplicate_seeds:
        violations.append({"check": "unique_seed_attack", "count": duplicate_seeds})
    seed_ids = {str(item.get("sample_id", "")) for item in seeds}
    held_out_seed_ids = seed_ids & (ids["validation"] | ids["test"])
    if held_out_seed_ids:
        violations.append({"check": "held_out_seed_id", "count": len(held_out_seed_ids)})

    file_paths = [seed_path, *paths.values()]
    return {
        "mode": mode_dir.name,
        "passed": not violations,
        "record_counts": {name: len(items) for name, items in records.items()},
        "unique_seed_count": len(seeds),
        "sha256": {path.name: sha256(path) for path in file_paths},
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate leakage-resistant paper splits")
    parser.add_argument("--root", default=str(dataset_root() / "co_agent_fuzz_paper"))
    parser.add_argument("--output", default=str(output_root() / "paper_split_validation.json"))
    args = parser.parse_args()
    root = Path(args.root)
    modes = [validate_mode(root / mode) for mode in ("hijacking", "extraction")]
    report = {"root": str(root.resolve()), "passed": all(item["passed"] for item in modes),
              "modes": modes}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
