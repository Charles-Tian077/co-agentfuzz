"""Offline rename checks using metadata and a fake subprocess only."""
import io
import unittest
from unittest.mock import patch

from Experiment.audit_evidence import label
from Experiment.compare_fuzzers import run_process
from Experiment.project_identity import canonical_method, identity


class ProjectIdentityTests(unittest.TestCase):
    def test_legacy_pairing_preserves_the_same_values(self):
        from Experiment.run_validation_matrix import paired_differences
        from tests.core.test_experiment_stats import _validation_row
        current = [_validation_row('PromptFuzz',1,100), _validation_row('Co-AgentFuzz',1,50)]
        legacy = [dict(row) for row in current]
        legacy[1]['method'] = identity()['method_aliases'][0]
        self.assertEqual(paired_differences(current), paired_differences(legacy))

    def test_every_historical_alias_maps_without_editing_evidence(self):
        for alias in identity()['method_aliases']:
            self.assertEqual(canonical_method(alias), 'Co-AgentFuzz')
            self.assertEqual(label({'method': alias}), 'Co-AgentFuzz')

    def test_baseline_and_ablation_variant_names_are_unchanged(self):
        for value in ('PromptFuzz','full','no_feedback_loop',None):
            self.assertEqual(canonical_method(value), value)

    def test_summary_marker_is_consistent_with_new_cli(self):
        class FakeProcess:
            stdout = io.StringIO('CO_AGENT_FUZZ_SUMMARY={"fixture": true}\n')
            def wait(self):
                return 0
        with patch('Experiment.compare_fuzzers.subprocess.Popen', return_value=FakeProcess()) as popen:
            self.assertEqual(run_process('Co-AgentFuzz', ['fixture-only']), {'fixture':True})
            popen.assert_called_once()


if __name__ == '__main__':
    unittest.main()
