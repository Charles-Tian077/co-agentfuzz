from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "item"


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_summary(command: List[str], show_child_output: bool) -> Dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    marker = "CO_AGENT_FUZZ_SUMMARY="
    summary = None
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if show_child_output:
            print(line, flush=True)
        if line.startswith(marker):
            summary = json.loads(line[len(marker):])
    if process.wait() != 0 or summary is None:
        raise RuntimeError("mutator tuning run failed or emitted no summary")
    return summary


def write_rows(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Co-AgentFuzz mutators on validation only")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--api-base", default="http://localhost:11434/v1")
    parser.add_argument("--modes", nargs="+", choices=["hijacking", "extraction"],
                        default=["hijacking", "extraction"])
    parser.add_argument("--defense-count", type=int, default=5)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--show-child-output", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(args.defense_count, args.queries, args.repeats, args.top_k) <= 0:
        raise ValueError("counts, queries, repeats, and top-k must be positive")

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    mutators = list(config["mutators"]["enabled"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    settings = vars(args)
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise RuntimeError("output directory contains a different tuning configuration")
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: List[Dict] = []

    selections = {}
    completed = 0
    total = len(args.modes) * len(mutators) * args.defense_count * args.repeats
    for offset, mode in enumerate(args.modes):
        dataset_dir = dataset_root() / "co_agent_fuzz_paper" / mode
        defenses_path = dataset_dir / "validation_defenses.jsonl"
        defenses = read_jsonl(defenses_path)
        count = min(args.defense_count, len(defenses))
        indices = sorted(random.Random(args.selection_seed + offset).sample(range(len(defenses)), count))
        selections[mode] = [{"defense_index": index,
                             "sample_id": defenses[index].get("sample_id")} for index in indices]
        for mutator in mutators:
            for defense_index in indices:
                for repeat in range(1, args.repeats + 1):
                    completed += 1
                    run_name = f"{mode}-{mutator}-{defense_index:04d}-r{repeat:02d}"
                    print(f"[{completed}/{total}] tune {run_name}", flush=True)
                    cell = (output / "cells" / mode / slug(mutator) /
                            f"defense-{defense_index:04d}" / f"repeat-{repeat:03d}")
                    summary_path = cell / "runner_summary.json"
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        print("  resumed", flush=True)
                    else:
                        cell.mkdir(parents=True, exist_ok=True)
                        command = [
                            sys.executable, "-m", "Experiment.run_co_agent_fuzz",
                            "--config", args.config, "--model", args.model,
                            "--api-base", args.api_base, "--max-queries", str(args.queries),
                            "--run-name", "run",
                            "--set", f"general.output_root={str(cell).replace(os.sep, '/')}",
                            "--set", f"general.random_seed={args.selection_seed + repeat}",
                            "--set", f"task.mode={mode}",
                            "--set", f"task.seeds_path={dataset_dir / 'pre_jailbreak_seeds.jsonl'}",
                            "--set", f"task.defenses_path={defenses_path}",
                            "--set", f"task.defense_index={defense_index}",
                            "--set", f"mutators.enabled=[{mutator}]",
                            "--set", f"general.verbose={str(args.show_child_output).lower()}",
                        ]
                        summary = run_summary(command, args.show_child_output)
                        summary_path.write_text(
                            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    rows.append({
                        "mode": mode, "mutator": mutator, "defense_index": defense_index,
                        "sample_id": defenses[defense_index].get("sample_id"), "repeat": repeat,
                        "strict_asr": summary["asr"], "hard_successes": summary["hard_successes"],
                        "target_queries": summary["target_queries"],
                        "first_success_query": summary["first_success_query"],
                        "variants_filtered": summary["variants_filtered"],
                        "total_llm_calls": summary["total_llm_calls"],
                        "total_tokens": summary.get("total_tokens", 0),
                        "wall_seconds": summary["wall_seconds"],
                    })
                    write_rows(output / "tuning_runs.csv", rows)
                    print(f"  ASR={float(summary['asr']):.3f} calls={summary['total_llm_calls']} "
                          f"tokens={summary.get('total_tokens', 0)}", flush=True)

    ranking = []
    for mutator in mutators:
        selected = [row for row in rows if row["mutator"] == mutator]
        total_successes = sum(int(row["hard_successes"]) for row in selected)
        total_queries = sum(int(row["target_queries"]) for row in selected)
        ranking.append({
            "mutator": mutator,
            "micro_strict_asr": total_successes / total_queries if total_queries else 0.0,
            "mean_run_asr": statistics.fmean(float(row["strict_asr"]) for row in selected),
            "any_success_rate": statistics.fmean(int(row["hard_successes"] > 0) for row in selected),
            "mean_wall_seconds": statistics.fmean(float(row["wall_seconds"]) for row in selected),
            "runs": len(selected),
        })
    ranking.sort(key=lambda row: (row["micro_strict_asr"], row["any_success_rate"],
                                  -row["mean_wall_seconds"]), reverse=True)
    chosen = [row["mutator"] for row in ranking[:min(args.top_k, len(ranking))]]
    result = {
        "selection_source": "validation_defenses_only",
        "selected_defenses": selections,
        "ranking": ranking,
        "selected_mutators": chosen,
        "override": "mutators.enabled=" + json.dumps(chosen, ensure_ascii=False),
    }
    (output / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nMUTATOR_SELECTION=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
