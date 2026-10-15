"""Arithmetic evidence tamper tests only; no model, network or attack modules."""

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from benign_orca import audit

from benign_orca.audit import AuditError, audit_records, audit_run, main
from benign_orca.benchmark import Config, run_matrix


class BenignAuditTests(unittest.TestCase):
    def setUp(self):
        self.summary = run_matrix(Config(train_episodes=8, eval_episodes=6, seeds=2))
        self.frames = self.summary.pop("sampled_traces")
        self.summary.update(run_id="11111111-1111-4111-8111-111111111111", source_sha256="a" * 64)
        for frame in self.frames:
            frame["run_id"] = self.summary["run_id"]

    def test_valid_run_and_actual_changes_are_distinct_from_calls(self):
        result = audit_records(self.summary, self.frames)
        self.assertEqual(result["trials"], 8)
        self.assertEqual(result["sampled_episodes_checked"], 32)
        for role, calls in result["sampled_learning_calls"].items():
            self.assertLessEqual(result["sampled_nonzero_value_changes"][role], calls)

    def test_summary_accounting_tampering_is_rejected(self):
        changes = {"episodes": 1, "llm_tokens": True, "llm_calls": 1, "scope": "other",
                   "summary_accuracy": {"full": 1}}
        for field, value in changes.items():
            with self.subTest(field=field):
                summary = copy.deepcopy(self.summary)
                summary[field] = value
                with self.assertRaises(AuditError):
                    audit_records(summary, self.frames)

    def test_missing_cost_is_not_silently_zero(self):
        self.summary.pop("llm_tokens")
        with self.assertRaises(KeyError):
            audit_records(self.summary, self.frames)

    def test_missing_sample_or_trial_is_rejected(self):
        with self.assertRaises(AuditError):
            audit_records(self.summary, self.frames[:-1])
        self.summary["runs"][-1] = copy.deepcopy(self.summary["runs"][0])
        with self.assertRaises(AuditError):
            audit_records(self.summary, self.frames)

    def test_trial_digest_update_count_and_freeze_tampering_is_rejected(self):
        for field, value in (("eval_task_sha256", "b" * 64), ("frozen_eval_verified", False),
                             ("policy_updates", dict.fromkeys(("generator", "refiner", "selector"), 0)),
                             ("feedback_receipts", 1), ("eval_accuracy", 1)):
            with self.subTest(field=field):
                summary = copy.deepcopy(self.summary)
                summary["runs"][0][field] = value
                with self.assertRaises(AuditError):
                    audit_records(summary, self.frames)

    def test_duplicate_identity_and_missing_receipt_are_rejected(self):
        for field, value in (("received", False), ("receipt_sequence", 2),
                             ("event_id", "duplicate"), ("verified_score", 1)):
            with self.subTest(field=field):
                frames = copy.deepcopy(self.frames)
                frames[0]["feedback"][0][field] = value
                with self.assertRaises(AuditError):
                    audit_records(self.summary, frames)
        self.frames[0]["feedback"].pop()
        with self.assertRaises(AuditError):
            audit_records(self.summary, self.frames)

    def test_reward_state_value_and_barrier_tampering_is_rejected(self):
        for field, value in (("reward", 1), ("peer_score", 1), ("state", "untrusted"),
                             ("new_value", 1), ("version_after", 0)):
            with self.subTest(field=field):
                frames = copy.deepcopy(self.frames)
                frames[0]["updates"][0][field] = value
                with self.assertRaises(AuditError):
                    audit_records(self.summary, frames)

    def test_orphan_reordered_or_duplicate_sample_is_rejected(self):
        for mutation in (lambda f: f[0].update(run_id="different"),
                         lambda f: f.reverse(),
                         lambda f: f.__setitem__(1, copy.deepcopy(f[0]))):
            frames = copy.deepcopy(self.frames)
            mutation(frames)
            with self.assertRaises(AuditError):
                audit_records(self.summary, frames)

    def test_consistent_but_false_sender_labels_are_rejected(self):
        for event in self.frames[0]["feedback"]:
            if event["sender"] == "generator":
                event["verified_score"] = 1.0
        with self.assertRaisesRegex(AuditError, "Arithmetic feedback label"):
            audit_records(self.summary, self.frames)

    def test_frozen_sample_cannot_mutate_values(self):
        frame = next(frame for frame in self.frames if frame["task_id"].startswith("eval/"))
        update = frame["updates"][0]
        update["new_value"] = 0 if update["old_value"] != 0 else 1
        with self.assertRaisesRegex(AuditError, "Value update equation"):
            audit_records(self.summary, self.frames)

    def test_zero_trace_budget_does_not_claim_sample_verification(self):
        summary = run_matrix(Config(train_episodes=4, eval_episodes=3, seeds=1, trace_episodes=0))
        summary.pop("sampled_traces")
        summary.update(run_id=self.summary["run_id"], source_sha256="a" * 64)
        result = audit_records(summary, [])
        self.assertEqual(result["sampled_episodes_checked"], 0)
        self.assertEqual(result["sampled_learning_calls"], {})

    def write_fixture(self, root, name, run_id):
        directory = root / name
        directory.mkdir()
        summary = {**self.summary, "run_id": run_id}
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (directory / "sampled_traces.jsonl").write_text("".join(
            json.dumps({**frame, "run_id": run_id}) + "\n" for frame in self.frames), encoding="utf-8")
        return directory

    def test_file_pairing_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.write_fixture(root, "服务器", self.summary["run_id"])
            second = self.write_fixture(root, "本机", "22222222-2222-4222-8222-222222222222")
            self.assertTrue(audit_run(first, second)["paired_local_results_match"])
            output = root / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            self.assertEqual(main([str(first), "--output", str(output)]), 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_mismatched_companion_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.write_fixture(root, "a", self.summary["run_id"])
            self.summary["source_sha256"] = "b" * 64
            second = self.write_fixture(root, "b", self.summary["run_id"])
            with self.assertRaises(AuditError):
                audit_run(first, second)

    def test_bad_json_cli_stops_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_fixture(root, "broken", self.summary["run_id"])
            (directory / "summary.json").write_text("not json", encoding="utf-8")
            output = root / "audit.json"
            self.assertEqual(main([str(directory), "--output", str(output)]), 2)
            self.assertFalse(output.exists())

    def test_auditor_imports_no_model_or_attack_modules(self):
        tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        allowed = {"__future__", "argparse", "hashlib", "json", "math", "random", "re", "sys",
                   "collections", "datetime", "pathlib", "uuid"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name.split(".")[0] in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split(".")[0], allowed)


if __name__ == "__main__":
    unittest.main()
