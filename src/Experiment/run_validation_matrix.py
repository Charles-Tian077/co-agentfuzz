from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    from .project_identity import canonical_method
except ImportError:
    from project_identity import canonical_method


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "model"


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_visible(command: List[str]) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"experiment command failed with exit code {process.returncode}")


def mean_stdev(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(values: List[float], seed: int = 20260904,
                      samples: int = 10000) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "n": len(values),
    }


def paired_randomisation_test(values: List[float], seed: int = 20260915,
                              samples: int = 100000) -> Dict[str, float]:
    """Two-sided paired sign-flip test for a mean effect."""
    if not values:
        return {"observed_mean": 0.0, "p_value": 1.0, "samples": 0}
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        draw = statistics.fmean(value if rng.random() < 0.5 else -value for value in values)
        extreme += int(abs(draw) >= observed - 1e-12)
    return {
        "observed_mean": statistics.fmean(values),
        "p_value": (extreme + 1) / (samples + 1),
        "samples": samples,
    }


def exact_mcnemar(coagent_only: int, promptfuzz_only: int) -> Dict[str, float]:
    discordant = coagent_only + promptfuzz_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(
            min(coagent_only, promptfuzz_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "coagent_only_successes": coagent_only,
        "promptfuzz_only_successes": promptfuzz_only,
        "discordant_units": discordant,
        "p_value": p_value,
    }


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values, returned in the original key space."""
    ordered = sorted((float(value), key) for key, value in p_values.items())
    total = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def paired_differences(rows: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
    grouped: Dict[tuple, Dict[str, Dict]] = {}
    for row in rows:
        key = (row["model"], row["mode"], row["defense_index"], row["repeat"])
        grouped.setdefault(key, {})[canonical_method(row["method"])] = row
    pairs = []
    for key, methods in grouped.items():
        if set(methods) != {"PromptFuzz", "Co-AgentFuzz"}:
            continue
        baseline = methods["PromptFuzz"]
        coagent = methods["Co-AgentFuzz"]
        baseline_calls = float(baseline["total_llm_calls"])
        first_baseline = baseline["first_success_query"] or int(baseline["target_queries"]) + 1
        first_coagent = coagent["first_success_query"] or int(coagent["target_queries"]) + 1
        baseline_tokens = float(baseline.get("total_tokens") or 0)
        coagent_tokens = float(coagent.get("total_tokens") or 0)
        pairs.append({
            "model": key[0], "mode": key[1], "defense_index": key[2], "repeat": key[3],
            "strict_asr_delta": float(coagent["strict_asr"]) - float(baseline["strict_asr"]),
            "llm_call_reduction": 1.0 - float(coagent["total_llm_calls"]) / baseline_calls
            if baseline_calls else 0.0,
            "token_reduction": (
                1.0 - coagent_tokens / baseline_tokens
                if baseline_tokens > 0 and coagent_tokens > 0 else None),
            "success_efficiency_delta": (
                float(coagent["successes_per_100_llm_calls"])
                - float(baseline["successes_per_100_llm_calls"])),
            "first_success_query_delta": float(first_coagent) - float(first_baseline),
            "coagent_success": int(coagent["hard_successes"] > 0),
            "promptfuzz_success": int(baseline["hard_successes"] > 0),
        })
    metrics = ("strict_asr_delta", "llm_call_reduction", "token_reduction",
               "success_efficiency_delta",
               "first_success_query_delta")
    run_level = {metric: bootstrap_mean_ci(
        [float(pair[metric]) for pair in pairs if pair[metric] is not None])
                 for metric in metrics}
    run_level["paired_runs"] = len(pairs)
    run_level["coagent_asr_win_rate"] = (
        sum(pair["strict_asr_delta"] > 0 for pair in pairs) / len(pairs) if pairs else 0.0)
    run_level["coagent_any_success_rate"] = (
        statistics.fmean(pair["coagent_success"] for pair in pairs) if pairs else 0.0)
    run_level["promptfuzz_any_success_rate"] = (
        statistics.fmean(pair["promptfuzz_success"] for pair in pairs) if pairs else 0.0)

    grouped_units: Dict[tuple, List[Dict]] = {}
    for pair in pairs:
        key = (pair["model"], pair["mode"], pair["defense_index"])
        grouped_units.setdefault(key, []).append(pair)
    unit_pairs = []
    for key, selected in grouped_units.items():
        unit_pairs.append({
            "model": key[0], "mode": key[1], "defense_index": key[2],
            **{
                metric: (statistics.fmean(float(item[metric]) for item in selected
                                          if item[metric] is not None)
                         if any(item[metric] is not None for item in selected) else None)
                for metric in metrics
            },
            "coagent_any_success": max(item["coagent_success"] for item in selected),
            "promptfuzz_any_success": max(item["promptfuzz_success"] for item in selected),
            "repeats": len(selected),
        })
    unit_level = {metric: bootstrap_mean_ci(
                    [float(unit[metric]) for unit in unit_pairs if unit[metric] is not None],
                                             seed=20260905)
                  for metric in metrics}
    unit_level["experimental_units"] = len(unit_pairs)
    unit_level["coagent_asr_win_rate"] = (
        sum(unit["strict_asr_delta"] > 0 for unit in unit_pairs) / len(unit_pairs)
        if unit_pairs else 0.0)
    strict_deltas = [float(unit["strict_asr_delta"]) for unit in unit_pairs]
    coagent_only = sum(
        unit["coagent_any_success"] and not unit["promptfuzz_any_success"]
        for unit in unit_pairs)
    promptfuzz_only = sum(
        unit["promptfuzz_any_success"] and not unit["coagent_any_success"]
        for unit in unit_pairs)
    significance = {
        "primary_unit": "model x task x defense (repeats averaged)",
        "strict_asr_paired_randomisation": paired_randomisation_test(strict_deltas),
        "any_success_exact_mcnemar": exact_mcnemar(coagent_only, promptfuzz_only),
        "alpha": 0.05,
    }
    return pairs, unit_pairs, {
        "unit_level": unit_level,
        "run_level": run_level,
        "significance": significance,
    }


def aggregate(rows: List[Dict]) -> Dict:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        result[model] = {}
        for mode in sorted({row["mode"] for row in rows if row["model"] == model}):
            result[model][mode] = {}
            for method in ("PromptFuzz", "Co-AgentFuzz"):
                selected = [row for row in rows if row["model"] == model
                            and row["mode"] == mode and canonical_method(row["method"]) == method]
                result[model][mode][method] = {}
                for metric in ("strict_asr", "target_error_rate", "total_llm_calls", "total_tokens", "wall_seconds",
                               "successes_per_100_llm_calls"):
                    values = [float(row[metric]) for row in selected
                              if row.get(metric) not in (None, "")]
                    if metric == "total_tokens":
                        values = [value for value in values if value > 0]
                    result[model][mode][method][metric] = mean_stdev(values)
    return result


def stratified_paired_summaries(rows: List[Dict], models: Iterable[str],
                                modes: Iterable[str]) -> Dict[str, Dict]:
    summaries: Dict[str, Dict] = {}
    for model in models:
        for mode in modes:
            key = f"{model}|{mode}"
            _, _, summaries[key] = paired_differences([
                row for row in rows
                if row["model"] == model and row["mode"] == mode
            ])
    return summaries


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable, paired Co-AgentFuzz validation matrix")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b"])
    parser.add_argument("--modes", nargs="+", choices=["hijacking", "extraction"],
                        default=["hijacking", "extraction"])
    parser.add_argument("--split", choices=["validation", "test"],
                        default="validation",
                        help="Evaluation split; use validation for tuning/pilots and test only after freezing")
    parser.add_argument("--defense-count", type=int, default=5)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--api-base", default="http://localhost:11434/v1")
    parser.add_argument("--target-max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coagent-set", action="append", default=[])
    args = parser.parse_args()
    if args.defense_count <= 0 or args.queries <= 0 or args.repeats <= 0:
        raise ValueError("defense-count, queries, and repeats must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    settings = vars(args)
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise RuntimeError("output directory contains a different experiment configuration")
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    python = sys.executable
    all_rows: List[Dict] = []
    selection = {}
    for mode_offset, mode in enumerate(args.modes):
        dataset_dir = dataset_root() / "co_agent_fuzz_paper" / mode
        seeds_file = dataset_dir / "pre_jailbreak_seeds.jsonl"
        defenses_file = dataset_dir / f"{args.split}_defenses.jsonl"
        defenses = read_jsonl(defenses_file)
        if args.defense_count > len(defenses):
            raise ValueError(
                f"requested {args.defense_count} {mode} defenses from {args.split}, "
                f"but only {len(defenses)} are available")
        count = args.defense_count
        rng = random.Random(args.selection_seed + mode_offset)
        indices = sorted(rng.sample(range(len(defenses)), count))
        selection[mode] = [{"defense_index": index,
                            "sample_id": defenses[index].get("sample_id")} for index in indices]

        for model in args.models:
            for position, defense_index in enumerate(indices, 1):
                cell = output / "cells" / slug(model) / mode / f"defense-{defense_index:04d}"
                report_path = cell / "comparison.json"
                print(f"\n### {model} | {mode} | defense {position}/{len(indices)} "
                      f"(index={defense_index}) ###", flush=True)
                if not report_path.exists():
                    command = [
                        python, "-m", "Experiment.compare_fuzzers",
                        "--mode", mode, "--phase", "focus",
                        "--defense-index", str(defense_index),
                        "--queries", str(args.queries), "--repeats", str(args.repeats),
                        "--model", model, "--api-base", args.api_base,
                        "--target-max-tokens", str(args.target_max_tokens),
                        "--seeds-file", str(seeds_file),
                        "--defenses-file", str(defenses_file),
                        "--output-dir", str(cell),
                    ]
                    for override in args.coagent_set:
                        command.extend(["--coagent-set", override])
                    run_visible(command)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                sample_id = defenses[defense_index].get("sample_id")
                for row in report["runs"]:
                    all_rows.append({"model": model, "mode": mode,
                                     "defense_index": defense_index, "sample_id": sample_id,
                                     **row})
                write_csv(output / "validation_runs.csv", all_rows)

    pairs, unit_pairs, paired_summary = paired_differences(all_rows)
    paired_by_mode = {}
    for mode in args.modes:
        _, _, mode_summary = paired_differences(
            [row for row in all_rows if row["mode"] == mode])
        paired_by_mode[mode] = mode_summary
    paired_summary["by_mode"] = paired_by_mode
    paired_by_model_task = stratified_paired_summaries(
        all_rows, args.models, args.modes)
    paired_summary["by_model_task"] = paired_by_model_task
    paired_summary["holm_corrected"] = {
        "strict_asr_paired_randomisation": holm_adjust({
            key: summary["significance"]["strict_asr_paired_randomisation"]["p_value"]
            for key, summary in paired_by_model_task.items()
        }),
        "any_success_exact_mcnemar": holm_adjust({
            key: summary["significance"]["any_success_exact_mcnemar"]["p_value"]
            for key, summary in paired_by_model_task.items()
        }),
    }
    write_csv(output / "paired_differences.csv", pairs)
    write_csv(output / "paired_unit_differences.csv", unit_pairs)
    summary = {
        "settings": settings,
        "selected_defenses": selection,
        "runs": len(all_rows),
        "aggregate": aggregate(all_rows),
        "paired_summary": paired_summary,
    }
    (output / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nVALIDATION_SUMMARY=" + json.dumps(paired_summary, ensure_ascii=False, sort_keys=True))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
