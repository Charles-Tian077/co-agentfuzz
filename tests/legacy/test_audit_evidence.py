"""Synthetic numerical fixtures only: not model-performance experiments."""

import unittest

from Experiment.audit_evidence import completeness, cost_effect, effect, holm, ledger, paired_units, validate_rows


def row(method, repeat=1, successes=3, **overrides):
    result = {"model": "fixture", "mode": "benign", "sample_id": "a",
              "method": method, "repeat": str(repeat), "random_seed": str(repeat),
              "target_queries": "10", "hard_successes": str(successes),
              "strict_asr": str(successes / 10), "total_tokens": ""}
    result.update(overrides)
    return result


class AuditTests(unittest.TestCase):
    def test_repeats_are_not_independent_defenses(self):
        rows = [row(method, repeat, successes) for repeat in (1, 2)
                for method, successes in (("A", 5), ("B", 3))]
        units, omitted = paired_units(rows, "A", "B")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["repeats"], 2)
        self.assertEqual(omitted, [])
        self.assertAlmostEqual(effect(units, 100)["difference"], .2)

    def test_missing_cost_does_not_become_zero(self):
        self.assertIsNone(ledger([row("A")])[0]["total_tokens"])

    def test_zero_cost_remains_a_measured_zero(self):
        self.assertEqual(ledger([row("A", total_tokens="0")])[0]["total_tokens"], 0)

    def test_duplicate_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([row("A"), row("A")])

    def test_invalid_denominator_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([row("A", target_queries="0")])

    def test_inconsistent_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([row("A", strict_asr=".9")])

    def test_incomplete_pair_is_not_silently_used(self):
        units, omitted = paired_units([row("A"), row("B"), row("A", 2)], "A", "B")
        self.assertEqual(units, [])
        self.assertEqual(len(omitted), 1)

    def test_budget_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_units([row("A"), row("B", target_queries="20")], "A", "B")

    def test_holm_adjustment_is_monotonic(self):
        adjusted = holm({"a": .01, "b": .04, "c": .2})
        self.assertAlmostEqual(adjusted["a"], .03)
        self.assertAlmostEqual(adjusted["b"], .08)
        self.assertAlmostEqual(adjusted["c"], .2)

    def test_no_effect_has_unit_p_value(self):
        units, _ = paired_units([row("A"), row("B")], "A", "B")
        self.assertEqual(effect(units, 100)["p_value"], 1)

    def test_missing_cost_prevents_efficiency_claim(self):
        self.assertFalse(cost_effect([row("A"), row("B")], "A", "B")["estimable"])

    def test_token_ratio_uses_totals_not_mean_of_ratios(self):
        rows = [row("A", total_tokens="10"), row("B", total_tokens="100"),
                row("A", sample_id="b", total_tokens="90"),
                row("B", sample_id="b", total_tokens="100")]
        self.assertAlmostEqual(cost_effect(rows, "A", "B", 100)["token_ratio"], .5)

    def test_absent_task_is_reported(self):
        settings = {"models": ["fixture"], "modes": ["benign", "absent"],
                    "defense_count": 1, "repeats": 1}
        result = completeness([row("A"), row("B")], settings)
        self.assertTrue(result[0]["complete"])
        self.assertFalse(result[1]["complete"])


if __name__ == "__main__":
    unittest.main()
