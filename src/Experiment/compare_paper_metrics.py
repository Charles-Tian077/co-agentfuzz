from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

from Experiment.compare_fuzzers import (  # noqa: E402
    available_run_name,
    run_process,
    write_json_atomic,
)
from Experiment.project_identity import canonical_method
from Experiment.run_validation_matrix import (  # noqa: E402
    bootstrap_mean_ci,
    holm_adjust,
    paired_randomisation_test,
)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "method"


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def paper_metrics(vectors: Sequence[Sequence[int]], defense_count: int) -> Dict[str, float]:
    if defense_count <= 0:
        raise ValueError("defense_count must be positive")
    normalised = [
        [int(bool(value)) for value in list(vector)[:defense_count]]
        + [0] * max(0, defense_count - len(vector))
        for vector in vectors
    ]
    ranked = sorted(normalised, key=sum, reverse=True)

    def union_rate(selected: Sequence[Sequence[int]]) -> float:
        return sum(any(vector[index] for vector in selected)
                   for index in range(defense_count)) / defense_count

    return {
        "best_asr": (sum(ranked[0]) / defense_count if ranked else 0.0),
        "esr_top5": union_rate(ranked[:5]) if ranked else 0.0,
        "coverage": union_rate(ranked) if ranked else 0.0,
        "evaluated_prompts": len(normalised),
        "successful_prompts": sum(sum(vector) > 0 for vector in normalised),
    }


def promptfuzz_vectors(path: Path, defense_count: int) -> List[List[int]]:
    vectors = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = ast.literal_eval(row.get("results", "[]"))
            if not isinstance(raw, list):
                raise ValueError(f"PromptFuzz result vector is not a list: {raw!r}")
            vectors.append([int(bool(value)) for value in raw[:defense_count]])
    return vectors


def coagent_vectors(path: Path, defense_ids: Sequence[str]) -> List[List[int]]:
    position = {str(defense_id): index for index, defense_id in enumerate(defense_ids)}
    vectors = []
    for candidate in read_jsonl(path):
        vector = [0] * len(defense_ids)
        for evaluation in candidate.get("evaluations", []):
            index = position.get(str(evaluation.get("defense_id")))
            if index is not None:
                oracle = evaluation.get("oracle", {})
                vector[index] = int(bool(oracle.get("benchmark_success", False)))
        if candidate.get("evaluations"):
            vectors.append(vector)
    return vectors


def complete_vectors(vectors: Sequence[Sequence[int]], expected: int,
                     defense_count: int, method: str) -> List[List[int]]:
    """Restore omitted all-zero vectors and reject incomplete accounting."""
    completed = [list(vector) for vector in vectors]
    if len(completed) > expected:
        raise RuntimeError(
            f"{method} emitted {len(completed)} prompt vectors for only {expected} expected")
    completed.extend([[0] * defense_count for _ in range(expected - len(completed))])
    return completed


def aggregate(rows: Sequence[Dict]) -> Dict:
    metrics = ("best_asr", "esr_top5", "coverage", "total_llm_calls",
               "total_tokens", "wall_seconds")
    output = {}
    for method in ("PromptFuzz", "Co-AgentFuzz"):
        selected = [row for row in rows if canonical_method(row["method"]) == method]
        output[method] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in selected
                      if metric != "total_tokens" or float(row[metric]) > 0]
            output[method][metric] = {
                "mean": statistics.fmean(values) if values else 0.0,
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
    return output


def paired_effects(rows: Sequence[Dict]) -> tuple[List[Dict], Dict]:
    """Summarise paired stochastic repeats over one fixed defense panel."""
    grouped: Dict[int, Dict[str, Dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["repeat"]), {})[canonical_method(row["method"])] = row
    pairs: List[Dict] = []
    for repeat, methods in sorted(grouped.items()):
        if set(methods) != {"PromptFuzz", "Co-AgentFuzz"}:
            continue
        baseline = methods["PromptFuzz"]
        coagent = methods["Co-AgentFuzz"]

        def reduction(metric: str):
            base = float(baseline[metric])
            new = float(coagent[metric])
            return 1.0 - new / base if base > 0 and new > 0 else None

        pairs.append({
            "repeat": repeat,
            "random_seed": int(coagent["random_seed"]),
            "best_asr_delta": float(coagent["best_asr"]) - float(baseline["best_asr"]),
            "esr_top5_delta": float(coagent["esr_top5"]) - float(baseline["esr_top5"]),
            "coverage_delta": float(coagent["coverage"]) - float(baseline["coverage"]),
            "llm_call_reduction": reduction("total_llm_calls"),
            "token_reduction": reduction("total_tokens"),
            "wall_time_reduction": reduction("wall_seconds"),
        })
    metrics = (
        "best_asr_delta", "esr_top5_delta", "coverage_delta",
        "llm_call_reduction", "token_reduction", "wall_time_reduction",
    )
    estimates = {
        metric: bootstrap_mean_ci([
            float(pair[metric]) for pair in pairs if pair[metric] is not None
        ], seed=20260915 + position)
        for position, metric in enumerate(metrics)
    }
    success_tests = {
        metric: paired_randomisation_test([
            float(pair[metric]) for pair in pairs if pair[metric] is not None
        ], seed=20260925 + position)
        for position, metric in enumerate(metrics[:3])
    }
    adjusted = holm_adjust({
        metric: result["p_value"] for metric, result in success_tests.items()
    })
    for metric, p_value in adjusted.items():
        success_tests[metric]["holm_p_value"] = p_value
    return pairs, {
        "primary_unit": "paired stochastic repeat over one fixed defense panel",
        "paired_repeats": len(pairs),
        "estimates": estimates,
        "success_metric_tests": success_tests,
        "alpha": 0.05,
        "inference_scope": (
            "random-seed variability for this fixed panel; defense-level generalisation "
            "is assessed by the validation matrix"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Equal-budget PromptFuzz comparison with original cross-defense metrics")
    parser.add_argument("--mode", choices=["hijacking", "extraction"], required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--api-base", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--seeds-file", required=True)
    parser.add_argument("--defenses-file", required=True)
    parser.add_argument("--seed-count", type=int, default=80)
    parser.add_argument("--defense-count", type=int, default=50)
    parser.add_argument("--queries", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260905)
    parser.add_argument("--target-max-tokens", type=int, default=128)
    parser.add_argument("--config", default=str(
        default_config()))
    parser.add_argument("--coagent-set", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(args.seed_count, args.defense_count, args.queries, args.repeats) <= 0:
        raise ValueError("counts, queries, and repeats must be positive")
    if args.queries % args.defense_count:
        raise ValueError("queries must be divisible by defense-count for complete prompt vectors")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args)
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise RuntimeError("output directory contains a different paper-metric configuration")
    if not settings_path.exists():
        write_json_atomic(settings_path, settings)

    all_seeds = read_jsonl(Path(args.seeds_file))
    all_defenses = read_jsonl(Path(args.defenses_file))
    if args.seed_count > len(all_seeds) or args.defense_count > len(all_defenses):
        raise ValueError("requested sample exceeds available seeds or defenses")
    seed_rng = random.Random(args.selection_seed)
    defense_rng = random.Random(args.selection_seed + 1)
    selected_seeds = [all_seeds[index] for index in sorted(
        seed_rng.sample(range(len(all_seeds)), args.seed_count))]
    selected_defenses = [all_defenses[index] for index in sorted(
        defense_rng.sample(range(len(all_defenses)), args.defense_count))]
    selected_seeds_path = output / "selected_seeds.jsonl"
    selected_defenses_path = output / "selected_defenses.jsonl"
    if not selected_seeds_path.exists():
        write_jsonl(selected_seeds_path, selected_seeds)
        write_jsonl(selected_defenses_path, selected_defenses)
    defense_ids = [str(item.get("sample_id", index))
                   for index, item in enumerate(selected_defenses)]
    if len(set(defense_ids)) != len(defense_ids):
        raise ValueError("selected defenses contain duplicate sample_id values")

    rows: List[Dict] = []
    for repeat in range(1, args.repeats + 1):
        random_seed = 20260904 + repeat
        baseline_csv = output / f"promptfuzz-{repeat:03d}.csv"
        coagent_root = output / "co_agent_fuzz"
        coagent_name = available_run_name(coagent_root, f"repeat-{repeat:03d}")
        commands = {
            "PromptFuzz": [
                sys.executable, "-m", "Experiment.run", "--phase", "focus",
                "--mode", args.mode, "--index", "0", "--all_defenses",
                "--model_path", args.model, "--api_base", args.api_base,
                "--max_query", str(args.queries), "--max_jailbreak", "999999999",
                "--strict_budget", "--random_seed", str(random_seed),
                "--result_file", str(baseline_csv),
                "--seed_file", str(selected_seeds_path),
                "--defense_file", str(selected_defenses_path),
                "--target_max_tokens", str(args.target_max_tokens),
            ],
            "Co-AgentFuzz": [
                sys.executable, "-m", "Experiment.run_co_agent_fuzz",
                "--config", args.config, "--model", args.model,
                "--api-base", args.api_base, "--max-queries", str(args.queries),
                "--max-iterations", str(max(100, args.queries)),
                "--run-name", coagent_name,
                "--set", f"general.random_seed={random_seed}",
                "--set", f"general.output_root={str(coagent_root).replace(os.sep, '/')}",
                "--set", f"task.mode={args.mode}",
                "--set", f"task.seeds_path={selected_seeds_path}",
                "--set", f"task.defenses_path={selected_defenses_path}",
                "--set", "task.all_defenses=true", "--set", "task.max_defenses=0",
                "--set", f"target.max_tokens={args.target_max_tokens}",
            ],
        }
        order = ["PromptFuzz", "Co-AgentFuzz"]
        if repeat % 2 == 0:
            order.reverse()
        summaries = {}
        for method in order:
            checkpoint = output / f"summary-{slug(method)}-{repeat:03d}.json"
            if checkpoint.exists():
                summaries[method] = json.loads(checkpoint.read_text(encoding="utf-8"))
                print(f"[{repeat}/{args.repeats}] {method}: resumed", flush=True)
            else:
                print(f"[{repeat}/{args.repeats}] {method}: running", flush=True)
                summary = run_process(method, commands[method])
                summary["random_seed"] = random_seed
                write_json_atomic(checkpoint, summary)
                summaries[method] = summary

        for method in ("PromptFuzz", "Co-AgentFuzz"):
            summary = summaries[method]
            target_queries = int(summary["target_queries"])
            if target_queries != args.queries:
                raise RuntimeError(
                    f"{method} used {target_queries} target queries; expected exactly {args.queries}")
            expected_vectors = args.queries // args.defense_count
            if method == "PromptFuzz":
                vectors = complete_vectors(
                    promptfuzz_vectors(baseline_csv, args.defense_count),
                    expected_vectors, args.defense_count, method)
            else:
                vectors = coagent_vectors(
                    Path(summary["output_dir"]) / "candidates.jsonl", defense_ids)
                if len(vectors) != expected_vectors:
                    raise RuntimeError(
                        f"{method} recorded {len(vectors)} evaluated prompts; "
                        f"expected {expected_vectors}")
            rows.append({
                "method": method,
                "repeat": repeat,
                "random_seed": random_seed,
                **paper_metrics(vectors, args.defense_count),
                "target_queries": target_queries,
                "target_errors": int(summary.get("target_errors", 0)),
                "target_error_rate": int(summary.get("target_errors", 0)) / target_queries,
                "hard_successes": int(summary["hard_successes"]),
                "total_llm_calls": int(summary["total_llm_calls"]),
                "total_tokens": int(summary.get("total_tokens", 0)),
                "wall_seconds": float(summary["wall_seconds"]),
                "artifact": summary.get("output_dir", summary.get("result_file")),
            })
        with (output / "paper_metrics.csv").open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    paired_rows, paired_summary = paired_effects(rows)
    if paired_rows:
        with (output / "paper_metric_paired_differences.csv").open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
            writer.writeheader()
            writer.writerows(paired_rows)
    report = {
        "settings": settings,
        "defense_ids": defense_ids,
        "runs": rows,
        "aggregate": aggregate(rows),
        "paired_differences": paired_rows,
        "paired_summary": paired_summary,
        "baseline_scope": (
            "PromptFuzz-compatible cold-start focus baseline; preparation-stage "
            "ranking/weights/few-shot artifacts were not regenerated"),
    }
    write_json_atomic(output / "paper_metrics.json", report)
    print("PAPER_METRICS=" + json.dumps(report["aggregate"], sort_keys=True))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
