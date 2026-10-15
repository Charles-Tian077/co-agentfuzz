"""Offline numerical audit. Never imports a fuzzer, queries a model, or trains.

Reads aggregate CSVs only; prompts, responses, keys and endpoints are not exported.
Repeated runs are averaged within a defense before statistical inference.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

try:
    from .project_identity import canonical_method
except ImportError:
    from project_identity import canonical_method


def load_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, field):
    value = row.get(field)
    if value is None or str(value).strip() == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}")
    return value


def label(row):
    name = row.get("variant") or row.get("method")
    return canonical_method(name)


def unit_key(row):
    # Row positions/defense indexes are not stable across different datasets.
    sample_id = row.get("sample_id")
    if not sample_id:
        raise ValueError("Missing stable sample_id")
    return row["model"], row["mode"], sample_id


def validate_rows(rows):
    issues, seen = [], set()
    for index, row in enumerate(rows, start=2):
        key = (*unit_key(row), label(row), row["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate experimental run at CSV line {index}")
        seen.add(key)
        queries = number(row, "target_queries")
        successes = number(row, "hard_successes")
        rate = number(row, "strict_asr")
        if queries is None or queries <= 0 or queries != int(queries):
            raise ValueError(f"Invalid query denominator at CSV line {index}")
        if successes is None or not 0 <= successes <= queries or successes != int(successes):
            raise ValueError(f"Invalid numerator at CSV line {index}")
        if rate is None or not math.isclose(rate, successes / queries, abs_tol=1e-9):
            raise ValueError(f"ASR does not match numerator/denominator at line {index}")
        tokens = [number(row, field) for field in
                  ("prompt_tokens", "completion_tokens", "total_tokens")]
        if any(value is not None and (value < 0 or value != int(value)) for value in tokens):
            raise ValueError(f"Invalid token count at line {index}")
        if all(value is not None for value in tokens) and tokens[0] + tokens[1] != tokens[2]:
            issues.append(f"Token components do not add up at line {index}")
    return issues


def paired_units(rows, reference, control):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if label(row) in {reference, control}:
            groups[unit_key(row)][label(row)][row["repeat"]] = row
    units, omitted = [], []
    for key, methods in sorted(groups.items()):
        left, right = methods[reference], methods[control]
        if not left or set(left) != set(right):
            omitted.append({"unit": key, "reason": "missing method or repeat"})
            continue
        repeats = sorted(left)
        for repeat in repeats:
            a, b = left[repeat], right[repeat]
            if a.get("random_seed") != b.get("random_seed"):
                raise ValueError("Paired random seeds differ")
            if number(a, "target_queries") != number(b, "target_queries"):
                raise ValueError("Paired query budgets differ")
        units.append({
            "model": key[0], "mode": key[1], "sample_id": key[2],
            "repeats": len(repeats),
            "reference_asr": mean(number(left[r], "strict_asr") for r in repeats),
            "control_asr": mean(number(right[r], "strict_asr") for r in repeats),
        })
    return units, omitted


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def effect(units, resamples=12000):
    if not units:
        return {"independent_defenses": 0, "estimable": False}
    diffs = [row["reference_asr"] - row["control_asr"] for row in units]
    strata = defaultdict(list)
    for row, value in zip(units, diffs):
        strata[(row["model"], row["mode"])].append(value)
    rng = random.Random(20260916)
    boot = [mean(value for group in strata.values()
                 for value in rng.choices(group, k=len(group))) for _ in range(resamples)]
    observed = abs(sum(diffs))
    if len(diffs) <= 16:
        p_value = sum(abs(sum(d * s for d, s in zip(diffs, signs))) >= observed - 1e-12
                      for signs in itertools.product((-1, 1), repeat=len(diffs))) / 2 ** len(diffs)
        test = "exact two-sided defense-level sign-flip"
    else:
        count = sum(abs(sum(d * rng.choice((-1, 1)) for d in diffs)) >= observed - 1e-12
                    for _ in range(resamples))
        p_value = (count + 1) / (resamples + 1)
        test = "Monte Carlo two-sided defense-level sign-flip"
    return {
        "estimable": True, "independent_defenses": len(units),
        "paired_repeat_runs": sum(row["repeats"] for row in units),
        "reference_asr": mean(row["reference_asr"] for row in units),
        "control_asr": mean(row["control_asr"] for row in units),
        "difference": mean(diffs), "ci95": [percentile(boot, .025), percentile(boot, .975)],
        "p_value": p_value, "test": test, "bootstrap_resamples": resamples,
    }


def holm(p_values):
    result, previous = {}, 0.0
    ordered = sorted(p_values, key=p_values.get)
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, p_values[key] * (len(ordered) - rank)))
        result[key] = previous
    return result


def ledger(rows):
    entries = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mode"], label(row))].append(row)
    for (model, mode, method), group in sorted(groups.items()):
        entry = {"model": model, "mode": mode, "method": method, "runs": len(group)}
        for field in ("hard_successes", "target_queries", "total_llm_calls", "total_tokens",
                      "prompt_tokens", "completion_tokens", "wall_seconds", "target_errors"):
            values = [number(row, field) for row in group]
            entry[field] = sum(values) if all(value is not None for value in values) else None
            entry[f"{field}_missing_runs"] = sum(value is None for value in values)
        entry["strict_asr"] = mean(number(row, "strict_asr") for row in group)
        entries.append(entry)
    return entries


def comparisons(rows, reference, control):
    units, omitted = paired_units(rows, reference, control)
    return {
        "reference": reference, "control": control,
        "overall": effect(units),
        "by_mode": {mode: effect([row for row in units if row["mode"] == mode])
                    for mode in sorted({row["mode"] for row in units})},
        "omitted_units": omitted, "units": units,
    }


def cost_effect(rows, reference, control, resamples=12000):
    units, _ = paired_units(rows, reference, control)
    if not units:
        return {"estimable": False, "reason": "no complete paired defenses"}
    indexed = defaultdict(list)
    for row in rows:
        indexed[(*unit_key(row), label(row))].append(row)
    costs = []
    for unit in units:
        key = (unit["model"], unit["mode"], unit["sample_id"])
        left = indexed[(*key, reference)]
        right = indexed[(*key, control)]
        values = [number(row, "total_tokens") for row in left + right]
        if any(value is None for value in values):
            return {"estimable": False, "reason": "total_tokens missing; never imputed as zero"}
        costs.append((unit["model"], unit["mode"], sum(values[:len(left)]), sum(values[len(left):])))
    denominator = sum(item[3] for item in costs)
    if denominator <= 0:
        return {"estimable": False, "reason": "zero control-token denominator"}
    groups = defaultdict(list)
    for item in costs:
        groups[item[:2]].append(item)
    rng, bootstrap = random.Random(20260916), []
    for _ in range(resamples):
        sample = [item for group in groups.values() for item in rng.choices(group, k=len(group))]
        base = sum(item[3] for item in sample)
        if base > 0:
            bootstrap.append(sum(item[2] for item in sample) / base)
    return {
        "estimable": True, "independent_defenses": len(units),
        "reference_tokens": sum(item[2] for item in costs), "control_tokens": denominator,
        "token_ratio": sum(item[2] for item in costs) / denominator,
        "ratio_ci95": [percentile(bootstrap, .025), percentile(bootstrap, .975)],
        "scope": "recorded run tokens only; historical training/tuning/setup not established",
    }


def source_inventory(root, local_path):
    files = sorted(Path(root).glob("*/*.*")) + [Path(local_path)]
    return [{"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]


def completeness(rows, settings):
    expected_groups = [(model, mode) for model in settings["models"] for mode in settings["modes"]]
    observed = defaultdict(int)
    for row in rows:
        observed[(row["model"], row["mode"])] += 1
    per_group = settings["defense_count"] * settings["repeats"] * 2
    return [{"model": model, "mode": mode, "expected_runs": per_group,
             "observed_runs": observed[(model, mode)],
             "complete": observed[(model, mode)] == per_group} for model, mode in expected_groups]


def sql_cross_checks(validation, ablation, local, root):
    """Independently recompute chart point estimates from the original CSVs."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        for name, rows in (("validation_csv", validation), ("ablation_csv", ablation),
                           ("local_csv", local)):
            connection.execute(f"CREATE TABLE {name} (model TEXT, mode TEXT, sample_id TEXT, "
                               "method TEXT, repeat TEXT, strict_asr REAL)")
            connection.executemany(f"INSERT INTO {name} VALUES (?, ?, ?, ?, ?, ?)",
                                   [(row["model"], row["mode"], row["sample_id"], label(row),
                                     row["repeat"], number(row, "strict_asr")) for row in rows])
        paper = []
        for mode in ("hijacking", "extraction"):
            for row in load_rows(Path(root) / f"paper-{mode}/paper_metrics.csv"):
                paper.append((mode, label(row), *[number(row, field) for field in
                              ("total_tokens", "total_llm_calls", "target_queries", "best_asr",
                               "esr_top5", "coverage", "wall_seconds")]))
        connection.execute("CREATE TABLE paper_csv (mode TEXT, method TEXT, total_tokens REAL, "
                           "total_llm_calls REAL, target_queries REAL, best_asr REAL, "
                           "esr_top5 REAL, coverage REAL, wall_seconds REAL)")
        connection.executemany("INSERT INTO paper_csv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", paper)
        query = """WITH defense AS (
  SELECT model, mode, sample_id, method, AVG(strict_asr) AS asr
  FROM {table} GROUP BY model, mode, sample_id, method
)
SELECT control.method AS control, COUNT(*) AS independent_defenses,
       AVG(full.asr) AS reference_asr, AVG(control.asr) AS control_asr,
       100 * AVG(full.asr - control.asr) AS delta_pp
FROM defense AS full JOIN defense AS control
  ON full.model = control.model AND full.mode = control.mode
 AND full.sample_id = control.sample_id
WHERE full.method = '{reference}' AND control.method {condition}
GROUP BY control.method ORDER BY control.method"""
        queries = {
            "ablation": query.format(table="ablation_csv", reference="full", condition="!= 'full'"),
            "wfu": query.format(table="validation_csv", reference="Co-AgentFuzz", condition="= 'PromptFuzz'"),
            "paper": "SELECT * FROM paper_csv ORDER BY mode, method",
        }
        return {name: {"sql": sql, "rows": [dict(row) for row in connection.execute(sql)]}
                for name, sql in queries.items()}
    finally:
        connection.close()


def run_audit(root, local_path):
    root = Path(root)
    validation = load_rows(root / "validation/validation_runs.csv")
    ablation = load_rows(root / "ablation/ablation_runs.csv")
    local = load_rows(local_path)
    with Path(local_path).with_name("settings.json").open(encoding="utf-8-sig") as handle:
        local_settings = json.load(handle)
    issues = {name: validate_rows(rows) for name, rows in
              (("wfu_validation", validation), ("wfu_ablation", ablation), ("local", local))}
    with (root / "ablation/ablation_artifact_audit.json").open(encoding="utf-8-sig") as handle:
        upstream = json.load(handle)
    verified = {}
    for filename, expected in upstream["sha256"].items():
        actual = hashlib.sha256((root / "ablation" / filename).read_bytes()).hexdigest()
        verified[filename] = actual == expected
        if actual != expected:
            raise ValueError(f"Upstream artifact hash mismatch: {filename}")
    variants = sorted({label(row) for row in ablation} - {"full"})
    ablations = {variant: comparisons(ablation, "full", variant) for variant in variants}
    adjusted = holm({variant: item["overall"]["p_value"] for variant, item in ablations.items()})
    for variant, item in ablations.items():
        item["overall"]["holm_p"] = adjusted[variant]
        item["recorded_cost"] = cost_effect(ablation, "full", variant)
    full_checks = [row for row in upstream["feedback_checks"] if row["variant"] == "full"]
    feedback = {field: sum(row[field] for row in full_checks) for field in
                ("iterations", "feedback_cycles", "feedback_seed_selections",
                 "evolved_seed_selections", "prompt_proposals", "selected_variants", "nonzero_global_q")}
    full_rows = [row for row in ablation if label(row) == "full"]
    feedback["full_runs"] = len(full_rows)
    feedback["full_target_queries"] = sum(number(row, "target_queries") for row in full_rows)
    paper = []
    for mode in ("hijacking", "extraction"):
        for row in load_rows(root / f"paper-{mode}/paper_metrics.csv"):
            paper.append({"mode": mode, "method": label(row), **{
                field: number(row, field) for field in
                ("best_asr", "esr_top5", "coverage", "target_queries", "total_llm_calls",
                 "total_tokens", "wall_seconds", "hard_successes", "evaluated_prompts")}})
    local_results = {}
    for model in sorted({row["model"] for row in local}):
        rows = [row for row in local if row["model"] == model]
        local_results[model] = comparisons(rows, "Co-AgentFuzz", "PromptFuzz")
    sql_checks = sql_cross_checks(validation, ablation, local, root)
    for row in sql_checks["ablation"]["rows"]:
        if not math.isclose(row["delta_pp"], 100 * ablations[row["control"]]["overall"]["difference"],
                            abs_tol=1e-9):
            raise ValueError("Independent SQL point-estimate check failed")
    with (root / "retrieval.json").open(encoding="utf-8-sig") as handle:
        retrieval = json.load(handle)
    server_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
                     for name, expected in retrieval["server_sha256"].items()}
    if not all(server_hashes.values()):
        raise ValueError("Downloaded evidence differs from server checksums")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "historical aggregate audit; no new inference, training, or performance experiment",
        "inference_unit": "model/task/stable sample_id; average paired repeats within defense",
        "bootstrap": "task/model-stratified defense bootstrap; 12000 draws; seed 20260916",
        "multiplicity": "Holm across six pooled ablations; task-specific/local checks exploratory",
        "sources": source_inventory(root, local_path), "issues": issues,
        "ablation_hash_checks": verified,
        "server_hash_checks": server_hashes, "sql_cross_checks": sql_checks,
        "wfu_comparison": comparisons(validation, "Co-AgentFuzz", "PromptFuzz"),
        "wfu_comparison_cost": ledger(validation),
        "wfu_recorded_token_effect": cost_effect(validation, "Co-AgentFuzz", "PromptFuzz"),
        "wfu_ablation": ablations, "wfu_ablation_cost": ledger(ablation),
        "feedback": feedback, "paper_metrics": paper,
        "local_comparisons": local_results, "local_cost": ledger(local),
        "local_matrix_completeness": completeness(local, local_settings),
        "missing_controls": ["joint no-learning/no-feedback group", "pure message-exchange intervention"],
        "limitations": [
            "WFU runs use validation, not a locked independent test.",
            "PromptFuzz-compatible cold-start focus baseline; full preparation not established.",
            "Local and WFU model/runtime/budget/version differ; do not pool their performance.",
            "Upstream audit attests raw response checks; this audit recomputes aggregate CSVs only.",
            "Cost ledger does not establish coverage of historical training/tuning/setup expenditure.",
            "Selected variants are not automatically target requests; their counts must be distinguished.",
            "Bootstrap intervals with five defenses per task are exploratory and low precision.",
            "Historical statistical checks are not new preregistered confirmatory claims.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--local-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(args.evidence_root, args.local_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "scope": report["scope"],
                      "wfu_comparison": report["wfu_comparison"]["overall"],
                      "feedback": report["feedback"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
