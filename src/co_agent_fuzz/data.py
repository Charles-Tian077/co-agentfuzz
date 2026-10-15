from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def read_jsonl(path: str | Path) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def load_experiment_data(seed_path: str | Path, defense_path: str | Path,
                         defense_index: int = 0, all_defenses: bool = False,
                         max_defenses: int = 0) -> Tuple[List[str], List[Dict]]:
    seed_records = read_jsonl(seed_path)
    seeds = [str(item["attack"]).strip() for item in seed_records if str(item.get("attack", "")).strip()]
    defenses = read_jsonl(defense_path)
    if not all_defenses:
        if defense_index < 0 or defense_index >= len(defenses):
            raise IndexError(f"Defense index {defense_index} outside 0..{len(defenses) - 1}")
        defenses = [defenses[defense_index]]
    elif max_defenses > 0:
        defenses = defenses[:max_defenses]
    return seeds, defenses
