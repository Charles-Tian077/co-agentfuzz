from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "method"


def run_process(label: str, command: List[str]) -> Dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        bufsize=1,
    )
    marker = ("CO_AGENT_FUZZ_SUMMARY=" if label == "Co-AgentFuzz"
              else f"{label.upper().replace('-', '_')}_SUMMARY=")
    summary = None
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        print(f"[{label}] {line}", flush=True)
        raw_line = line
        if raw_line.startswith(marker):
            summary = json.loads(raw_line[len(marker):])
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} exited with code {return_code}")
    if summary is None:
        raise RuntimeError(f"{label} did not emit {marker}")
    return summary


def write_json_atomic(path: Path, payload: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def available_run_name(root: Path, base: str) -> str:
    if not (root / base).exists():
        return base
    attempt = 2
    while (root / f"{base}-attempt-{attempt:02d}").exists():
        attempt += 1
    return f"{base}-attempt-{attempt:02d}"


def metric_row(method: str, repeat: int, summary: Dict) -> Dict:
    target_queries = int(summary["target_queries"])
    hard_successes = int(summary["hard_successes"])
    llm_calls = int(summary["total_llm_calls"])
    target_errors = int(summary.get("target_errors", 0))
    return {
        "method": method,
        "repeat": repeat,
        "random_seed": summary.get("random_seed", 0),
        "strict_asr": hard_successes / target_queries if target_queries else 0.0,
        "oracle_asr": summary.get("oracle_asr"),
        "hard_successes": hard_successes,
        "target_queries": target_queries,
        "target_errors": target_errors,
        "target_error_rate": target_errors / target_queries if target_queries else 0.0,
        "total_llm_calls": llm_calls,
        "transport_requests": int(summary.get("transport_requests", llm_calls)),
        "prompt_tokens": int(summary.get("prompt_tokens", 0)),
        "completion_tokens": int(summary.get("completion_tokens", 0)),
        "total_tokens": int(summary.get("total_tokens", 0)),
        "auxiliary_llm_calls": llm_calls - target_queries,
        "first_success_query": summary.get("first_success_query"),
        "wall_seconds": float(summary["wall_seconds"]),
        "successes_per_100_llm_calls": 100.0 * hard_successes / llm_calls if llm_calls else 0.0,
        "variants_filtered": summary.get("variants_filtered"),
        "variants_generated": summary.get("variants_generated"),
        "output_dir": summary.get("output_dir", summary.get("result_file")),
    }


def aggregate(rows: List[Dict]) -> Dict:
    result = {}
    for method in ("PromptFuzz", "Co-AgentFuzz"):
        selected = [row for row in rows if canonical_method(row["method"]) == method]
        result[method] = {}
        for key in ("strict_asr", "target_error_rate", "total_llm_calls", "total_tokens", "wall_seconds",
                    "successes_per_100_llm_calls"):
            values = [float(row[key]) for row in selected]
            result[method][key] = {
                "mean": statistics.fmean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    base = result["PromptFuzz"]
    new = result["Co-AgentFuzz"]
    result["delta"] = {
        "strict_asr": new["strict_asr"]["mean"] - base["strict_asr"]["mean"],
        "llm_call_reduction_fraction": (
            1.0 - new["total_llm_calls"]["mean"] / base["total_llm_calls"]["mean"]
            if base["total_llm_calls"]["mean"] else 0.0),
        "success_efficiency_delta": (
            new["successes_per_100_llm_calls"]["mean"]
            - base["successes_per_100_llm_calls"]["mean"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible, equal-budget PromptFuzz comparison")
    parser.add_argument("--mode", choices=["hijacking", "extraction"], default="hijacking")
    parser.add_argument("--phase", choices=["focus", "preparation"], default="focus")
    parser.add_argument("--defense-index", type=int, default=0)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--api-base", default="http://localhost:11434/v1")
    parser.add_argument("--target-max-tokens", type=int, default=512,
                        help="Shared target-response token cap")
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--seeds-file")
    parser.add_argument("--defenses-file")
    parser.add_argument("--coagent-set", action="append", default=[])
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.queries <= 0 or args.repeats <= 0:
        raise ValueError("queries and repeats must be positive")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output_dir) if args.output_dir else output_root() / "comparisons" / timestamp
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    settings = vars(args)
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous != settings:
            raise RuntimeError("output directory contains a different comparison configuration")
    else:
        write_json_atomic(settings_path, settings)
    python = sys.executable
    rows: List[Dict] = []

    seeds_file = args.seeds_file or f"Datasets/{args.mode}_{args.phase}_seed.jsonl"
    defenses_file = args.defenses_file or f"Datasets/{args.mode}_{args.phase}_defense.jsonl"
    for repeat in range(1, args.repeats + 1):
        random_seed = 20260904 + repeat
        coagent_root = output / "co_agent_fuzz"
        coagent_run_name = available_run_name(coagent_root, f"repeat-{repeat:03d}")
        baseline_command = [
            python, "-m", "Experiment.run", "--phase", args.phase, "--mode", args.mode,
            "--index", str(args.defense_index), "--model_path", args.model,
            "--api_base", args.api_base, "--max_query", str(args.queries),
            "--max_jailbreak", "999999999", "--strict_budget",
            "--random_seed", str(random_seed), "--result_file",
            str(output / f"promptfuzz-{repeat:03d}.csv"),
            "--seed_file", seeds_file, "--defense_file", defenses_file,
            "--target_max_tokens", str(args.target_max_tokens),
        ]
        coagent_command = [
            python, "-m", "Experiment.run_co_agent_fuzz", "--config", args.config,
            "--model", args.model, "--api-base", args.api_base,
            "--max-queries", str(args.queries), "--run-name", coagent_run_name,
            "--set", f"general.random_seed={random_seed}",
            "--set", f"general.output_root={str(output / 'co_agent_fuzz').replace(os.sep, '/')}",
            "--set", f"task.mode={args.mode}",
            "--set", f"task.seeds_path={seeds_file}",
            "--set", f"task.defenses_path={defenses_file}",
            "--set", f"task.defense_index={args.defense_index}",
            "--set", f"target.max_tokens={args.target_max_tokens}",
        ]
        for override in args.coagent_set:
            coagent_command.extend(["--set", override])
        commands = [("PromptFuzz", baseline_command), ("Co-AgentFuzz", coagent_command)]
        if repeat % 2 == 0:
            commands.reverse()
        for method, command in commands:
            checkpoint = output / f"summary-{slug(method)}-{repeat:03d}.json"
            if checkpoint.exists():
                summary = json.loads(checkpoint.read_text(encoding="utf-8"))
                print(f"\n=== repeat {repeat}/{args.repeats}: {method} (resumed) ===",
                      flush=True)
            else:
                print(f"\n=== repeat {repeat}/{args.repeats}: {method} ===", flush=True)
                summary = run_process(method, command)
                summary["random_seed"] = random_seed
                write_json_atomic(checkpoint, summary)
            rows.append(metric_row(method, repeat, summary))

        with (output / "runs.partial.csv").open(
                "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fieldnames = list(rows[0])
    with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {"settings": vars(args), "runs": rows, "aggregate": aggregate(rows)}
    write_json_atomic(output / "comparison.json", report)
    print("\nCOMPARISON=" + json.dumps(report["aggregate"], ensure_ascii=False, sort_keys=True))
    print(f"Artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
