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
from typing import Dict, Iterable, List


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")


VARIANTS = {
    "full": [],
    "slow_rl": [
        "rl.learning_rate=0.10",
        "rl.epsilon_decay=0.995",
        "rl.global_credit_weight=0.35",
    ],
    "no_rl_policy": [
        "rl.epsilon_start=1.0",
        "rl.epsilon_end=1.0",
        "rl.epsilon_decay=1.0",
        "rl.global_credit_weight=0.0",
        "rl.dense_reward_weight=0.0",
        "rl.similarity_credit=0.0",
        "collaboration.memory_bias_weight=0.0",
        "collaboration.surrogate_bias_weight=0.0",
    ],
    "no_feedback_loop": [
        "collaboration.memory.enabled=false",
        "collaboration.memory_bias_weight=0.0",
        "collaboration.prompt_oversample_factor=1",
        "rl.dense_reward_weight=0.0",
        "rl.similarity_credit=0.0",
        "rl.global_credit_weight=0.0",
        "seed_pool.feedback_exploitation_rate=0.0",
        "surrogate.enabled=false",
    ],
    "legacy_core": [
        "collaboration.memory.enabled=false",
        "collaboration.no_repeat_window=0",
        "collaboration.selection_policy=epsilon_greedy",
        "collaboration.memory_bias_weight=0.0",
        "rl.dense_reward_weight=0.0",
        "rl.similarity_credit=0.0",
        "rl.global_credit_weight=0.0",
        "seed_pool.feedback_exploitation_rate=0.0",
        "collaboration.prompt_oversample_factor=1",
        "mutators.enabled=[unicode_confusable,zero_width_insertion,token_swap,synonym_replacement,instruction_injection,sentence_reorder,role_play,contextualization,multilingual_wrapper,base64_wrapper,genetic_crossover]",
    ],
    "no_memory": [
        "collaboration.memory.enabled=false",
        "collaboration.memory_bias_weight=0.0",
    ],
    "no_action_mask": ["collaboration.no_repeat_window=0"],
    "no_dense_feedback": ["rl.dense_reward_weight=0.0"],
    "no_similarity_transfer": ["rl.similarity_credit=0.0"],
    "no_global_q": ["rl.global_credit_weight=0.0"],
    "no_seed_feedback": ["seed_pool.feedback_exploitation_rate=0.0"],
    "no_prompt_feedback": [
        "collaboration.prompt_oversample_factor=1",
        "surrogate.enabled=false",
    ],
    "no_linguistic_mutation": [
        "mutators.enabled=[unicode_confusable,zero_width_insertion,token_swap,synonym_replacement,instruction_injection,sentence_reorder,role_play,contextualization,multilingual_wrapper,base64_wrapper,genetic_crossover]",
    ],
    "random_scheduler": [
        "rl.epsilon_start=1.0",
        "rl.epsilon_end=1.0",
        "rl.epsilon_decay=1.0",
        "rl.global_credit_weight=0.0",
        "collaboration.memory_bias_weight=0.0",
        "collaboration.surrogate_bias_weight=0.0",
        "collaboration.no_repeat_window=0",
    ],
    "no_prefilter": ["filter.enabled=false"],
    "no_surrogate": ["surrogate.enabled=false"],
    "no_expansion": ["expand.min_score=2.0"],
}


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "model"


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_process(command: List[str], show_child_output: bool) -> Dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    summary = None
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if show_child_output:
            print(f"[Co-AgentFuzz] {line}", flush=True)
        if line.startswith("CO_AGENT_FUZZ_SUMMARY="):
            summary = json.loads(line.split("=", 1)[1])
    if process.wait() != 0:
        raise RuntimeError(f"Co-AgentFuzz exited with code {process.returncode}")
    if summary is None:
        raise RuntimeError("Co-AgentFuzz did not emit CO_AGENT_FUZZ_SUMMARY")
    return summary


def percentile(values: List[float], fraction: float) -> float:
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
    draws = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)]
    return {
        "mean": statistics.fmean(values),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
        "n": len(values),
    }


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


def result_row(model: str, mode: str, defense_index: int, sample_id: object,
               variant: str, repeat: int, summary: Dict) -> Dict:
    queries = int(summary["target_queries"])
    successes = int(summary["hard_successes"])
    generated = int(summary.get("variants_generated", 0))
    filtered = int(summary.get("variants_filtered", 0))
    return {
        "model": model,
        "mode": mode,
        "defense_index": defense_index,
        "sample_id": sample_id,
        "variant": variant,
        "repeat": repeat,
        "random_seed": summary.get("random_seed"),
        "strict_asr": successes / queries if queries else 0.0,
        "hard_successes": successes,
        "target_queries": queries,
        "target_errors": int(summary.get("target_errors", 0)),
        "target_error_rate": int(summary.get("target_errors", 0)) / queries if queries else 0.0,
        "total_llm_calls": int(summary["total_llm_calls"]),
        "prompt_tokens": int(summary.get("prompt_tokens", 0)),
        "completion_tokens": int(summary.get("completion_tokens", 0)),
        "total_tokens": int(summary.get("total_tokens", 0)),
        "wall_seconds": float(summary["wall_seconds"]),
        "first_success_query": summary.get("first_success_query"),
        "successes_per_100_queries": 100.0 * successes / queries if queries else 0.0,
        "variants_generated": generated,
        "variants_filtered": filtered,
        "filter_rate": filtered / generated if generated else 0.0,
        "iterations": int(summary.get("iterations", 0)),
        "seed_pool_size": int(summary.get("seed_pool_size", 0)),
        "output_dir": summary.get("output_dir"),
    }


def aggregate(rows: List[Dict], variants: Iterable[str]) -> Dict:
    metrics = (
        "strict_asr", "target_queries", "target_error_rate", "total_llm_calls", "total_tokens",
        "wall_seconds", "successes_per_100_queries", "filter_rate", "seed_pool_size",
    )
    result: Dict[str, Dict] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            metric: {
                "mean": statistics.fmean(float(row[metric]) for row in selected),
                "stdev": (statistics.stdev(float(row[metric]) for row in selected)
                          if len(selected) > 1 else 0.0),
                "n": len(selected),
            }
            for metric in metrics
        }
    return result


def paired_unit_effects(rows: List[Dict], variants: Iterable[str]) -> tuple[List[Dict], Dict]:
    grouped: Dict[tuple, Dict[str, List[Dict]]] = {}
    for row in rows:
        key = (row["model"], row["mode"], row["defense_index"])
        grouped.setdefault(key, {}).setdefault(row["variant"], []).append(row)
    metrics = ("strict_asr", "target_queries", "total_tokens", "wall_seconds")
    units: List[Dict] = []
    for key, by_variant in grouped.items():
        if "full" not in by_variant:
            continue
        full = {metric: statistics.fmean(float(item[metric]) for item in by_variant["full"])
                for metric in metrics}
        for variant in variants:
            if variant == "full" or variant not in by_variant:
                continue
            ablated = {metric: statistics.fmean(float(item[metric]) for item in by_variant[variant])
                       for metric in metrics}
            units.append({
                "model": key[0], "mode": key[1], "defense_index": key[2],
                "variant": variant,
                **{f"{metric}_delta_vs_full": ablated[metric] - full[metric]
                   for metric in metrics},
                "repeats": len(by_variant[variant]),
            })
    summary: Dict[str, Dict] = {}
    for variant in variants:
        if variant == "full":
            continue
        selected = [unit for unit in units if unit["variant"] == variant]
        summary[variant] = {
            f"{metric}_delta_vs_full": bootstrap_mean_ci(
                [float(unit[f"{metric}_delta_vs_full"]) for unit in selected],
                seed=20260904 + position,
            )
            for position, metric in enumerate(metrics)
        }
    return units, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable Co-AgentFuzz ablations")
    parser.add_argument("--models", nargs="+", default=["qwen2.5:7b"])
    parser.add_argument("--modes", nargs="+", choices=["hijacking", "extraction"],
                        default=["hijacking", "extraction"])
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--defense-count", type=int, default=3)
    parser.add_argument("--budget-mode", choices=["queries", "iterations"], default="queries")
    parser.add_argument("--budget", type=int, default=45)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260904)
    parser.add_argument("--api-base", default="http://localhost:11434/v1")
    parser.add_argument("--target-max-tokens", type=int, default=128)
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--coagent-set", action="append", default=[])
    parser.add_argument("--show-child-output", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.defense_count <= 0 or args.budget <= 0 or args.repeats <= 0:
        raise ValueError("defense-count, budget, and repeats must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = vars(args)
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise RuntimeError("output directory contains a different experiment configuration")
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: List[Dict] = []
    selection: Dict[str, List[Dict]] = {}
    completed = 0
    total = len(args.models) * len(args.modes) * args.defense_count * len(args.variants) * args.repeats
    for mode_offset, mode in enumerate(args.modes):
        data_dir = dataset_root() / "co_agent_fuzz_paper" / mode
        seeds_file = data_dir / "pre_jailbreak_seeds.jsonl"
        defenses_file = data_dir / f"{args.split}_defenses.jsonl"
        defenses = read_jsonl(defenses_file)
        if args.defense_count > len(defenses):
            raise ValueError(
                f"requested {args.defense_count} {mode} defenses from {args.split}, "
                f"but only {len(defenses)} are available")
        count = args.defense_count
        indices = sorted(random.Random(args.selection_seed + mode_offset).sample(range(len(defenses)), count))
        selection[mode] = [{"defense_index": index, "sample_id": defenses[index].get("sample_id")}
                           for index in indices]
        for model in args.models:
            for defense_index in indices:
                sample_id = defenses[defense_index].get("sample_id")
                for variant in args.variants:
                    for repeat in range(1, args.repeats + 1):
                        completed += 1
                        cell = (output / "cells" / slug(model) / mode /
                                f"defense-{defense_index:04d}" / variant / f"repeat-{repeat:03d}")
                        runner_summary = cell / "runner_summary.json"
                        print(f"[{completed}/{total}] {model} | {mode} | defense={defense_index} | "
                              f"{variant} | repeat={repeat}", flush=True)
                        if runner_summary.exists():
                            summary = json.loads(runner_summary.read_text(encoding="utf-8"))
                            print("  resumed", flush=True)
                        else:
                            cell.mkdir(parents=True, exist_ok=True)
                            random_seed = 20260904 + repeat
                            command = [
                                sys.executable, "-m", "Experiment.run_co_agent_fuzz",
                                "--config", args.config, "--model", model,
                                "--api-base", args.api_base, "--run-name", "run",
                                "--set", f"general.random_seed={random_seed}",
                                "--set", f"general.output_root={str(cell).replace(os.sep, '/')}",
                                "--set", f"task.mode={mode}",
                                "--set", f"task.seeds_path={seeds_file}",
                                "--set", f"task.defenses_path={defenses_file}",
                                "--set", f"task.defense_index={defense_index}",
                                "--set", f"target.max_tokens={args.target_max_tokens}",
                                "--set", f"general.verbose={str(args.show_child_output).lower()}",
                            ]
                            if args.budget_mode == "queries":
                                command.extend(["--max-queries", str(args.budget),
                                                "--max-iterations", str(max(args.budget, 100))])
                            else:
                                command.extend(["--max-iterations", str(args.budget),
                                                "--max-queries", str(args.budget * 10)])
                            for override in [*VARIANTS[variant], *args.coagent_set]:
                                command.extend(["--set", override])
                            summary = run_process(command, args.show_child_output)
                            summary["random_seed"] = random_seed
                            runner_summary.write_text(
                                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                        row = result_row(model, mode, defense_index, sample_id,
                                         variant, repeat, summary)
                        rows.append(row)
                        write_csv(output / "ablation_runs.csv", rows)
                        print(f"  ASR={row['strict_asr']:.3f} queries={row['target_queries']} "
                              f"tokens={row['total_tokens']} time={row['wall_seconds']:.1f}s", flush=True)

    units, effects = paired_unit_effects(rows, args.variants)
    write_csv(output / "paired_unit_effects.csv", units)
    report = {
        "settings": settings,
        "variant_overrides": {name: VARIANTS[name] for name in args.variants},
        "selected_defenses": selection,
        "runs": len(rows),
        "aggregate": aggregate(rows, args.variants),
        "paired_unit_effects": effects,
    }
    (output / "ablation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ABLATION_SUMMARY=" + json.dumps(effects, ensure_ascii=False, sort_keys=True))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
