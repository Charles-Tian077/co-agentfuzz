from __future__ import annotations

from Experiment.run_ablation_matrix import paired_unit_effects
from Experiment.run_validation_matrix import (
    aggregate, holm_adjust, paired_differences, stratified_paired_summaries,
)
from Experiment.compare_paper_metrics import complete_vectors, paired_effects, paper_metrics
from Experiment.compare_fuzzers import slug
from Experiment.audit_validation_artifacts import strict_success


def _validation_row(method: str, repeat: int, tokens: int) -> dict:
    return {
        "model": "model-a",
        "mode": "hijacking",
        "defense_index": 1,
        "repeat": repeat,
        "method": method,
        "strict_asr": 0.5 if method == "Co-AgentFuzz" else 0.25,
        "total_llm_calls": 10 if method == "Co-AgentFuzz" else 20,
        "total_tokens": tokens,
        "wall_seconds": 1.0,
        "successes_per_100_llm_calls": 5.0,
        "first_success_query": 2,
        "target_queries": 10,
        "hard_successes": 1,
    }


def test_missing_token_counts_are_not_treated_as_zero_cost() -> None:
    rows = [_validation_row("PromptFuzz", 1, 0),
            _validation_row("Co-AgentFuzz", 1, 0)]
    pairs, units, summary = paired_differences(rows)
    assert pairs[0]["token_reduction"] is None
    assert units[0]["token_reduction"] is None
    assert summary["unit_level"]["token_reduction"]["n"] == 0
    assert aggregate(rows)["model-a"]["hijacking"]["PromptFuzz"]["total_tokens"]["n"] == 0


def test_ablation_effects_average_repeats_within_unit() -> None:
    rows = []
    for repeat, full_asr, ablated_asr in ((1, 0.5, 0.2), (2, 0.7, 0.4)):
        for variant, asr in (("full", full_asr), ("no_surrogate", ablated_asr)):
            rows.append({
                "model": "model-a", "mode": "hijacking", "defense_index": 1,
                "repeat": repeat, "variant": variant, "strict_asr": asr,
                "target_queries": 10, "total_tokens": 100, "wall_seconds": 1.0,
            })
    units, summary = paired_unit_effects(rows, ["full", "no_surrogate"])
    assert len(units) == 1
    assert abs(units[0]["strict_asr_delta_vs_full"] + 0.3) < 1e-12
    assert summary["no_surrogate"]["strict_asr_delta_vs_full"]["n"] == 1


def test_paper_metrics_use_best_prompt_top_five_union_and_full_coverage() -> None:
    vectors = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    metrics = paper_metrics(vectors, defense_count=4)
    assert metrics["best_asr"] == 0.5
    assert metrics["esr_top5"] == 1.0
    assert metrics["coverage"] == 1.0


def test_paper_metrics_count_missing_evaluations_as_failures() -> None:
    metrics = paper_metrics([[1, 1]], defense_count=4)
    assert metrics["best_asr"] == 0.5


def test_holm_adjustment_is_monotone_and_preserves_keys() -> None:
    adjusted = holm_adjust({"hijacking": 0.01, "extraction": 0.04, "third": 0.03})
    assert set(adjusted) == {"hijacking", "extraction", "third"}
    assert abs(adjusted["hijacking"] - 0.03) < 1e-12
    assert abs(adjusted["third"] - 0.06) < 1e-12
    assert abs(adjusted["extraction"] - 0.06) < 1e-12


def test_promptfuzz_omitted_failures_are_restored_as_zero_vectors() -> None:
    vectors = complete_vectors([[1, 0]], expected=3, defense_count=2,
                               method="PromptFuzz")
    assert vectors == [[1, 0], [0, 0], [0, 0]]
    assert paper_metrics(vectors, defense_count=2)["evaluated_prompts"] == 3


def test_paper_metric_effects_are_paired_by_repeat() -> None:
    rows = []
    for repeat, baseline, coagent in ((1, 0.25, 0.50), (2, 0.50, 0.75)):
        for method, value, calls in (
                ("PromptFuzz", baseline, 20), ("Co-AgentFuzz", coagent, 10)):
            rows.append({
                "method": method, "repeat": repeat, "random_seed": 7 + repeat,
                "best_asr": value, "esr_top5": value, "coverage": value,
                "total_llm_calls": calls, "total_tokens": calls * 10,
                "wall_seconds": float(calls),
            })
    pairs, summary = paired_effects(rows)
    assert len(pairs) == 2
    assert summary["paired_repeats"] == 2
    assert abs(summary["estimates"]["best_asr_delta"]["mean"] - 0.25) < 1e-12
    assert abs(summary["estimates"]["llm_call_reduction"]["mean"] - 0.5) < 1e-12


def test_comparison_checkpoint_slug_is_portable() -> None:
    assert slug("Co-AgentFuzz") == "Co-AgentFuzz"
    assert slug("  方法 / A  ") == "A"


def test_confirmatory_summaries_separate_model_task_hypotheses() -> None:
    rows = []
    for model in ("model-a", "model-b"):
        for mode in ("hijacking", "extraction"):
            for method in ("PromptFuzz", "Co-AgentFuzz"):
                row = _validation_row(method, repeat=1, tokens=100)
                row.update(model=model, mode=mode)
                rows.append(row)
    summaries = stratified_paired_summaries(
        rows, ["model-a", "model-b"], ["hijacking", "extraction"])
    assert set(summaries) == {
        "model-a|hijacking", "model-a|extraction",
        "model-b|hijacking", "model-b|extraction",
    }
    assert all(item["unit_level"]["experimental_units"] == 1
               for item in summaries.values())


def test_artifact_auditor_recomputes_strict_oracle() -> None:
    assert strict_success("[Access Granted]", "hijacking", "")
    assert not strict_success("System Access Granted", "hijacking", "")
    assert strict_success("prefix SeCrEt suffix", "extraction", "secret")
