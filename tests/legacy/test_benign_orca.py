"""Fixed arithmetic fixtures only: no models, prompts, network or attack code."""

import ast
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from benign_orca import benchmark

from benign_orca.benchmark import (
    ArithmeticTeam, Config, FuseTripped, Guard, Policy, Task,
    arithmetic_rule, main, run_matrix, tasks,
)


class BenignFeedbackTests(unittest.TestCase):
    def test_arithmetic_is_fixed_and_verifiable(self):
        for operation, expected in (("add", 11), ("subtract", 3), ("multiply", 28)):
            task = Task("fixture", 7, 4, operation)
            self.assertEqual(task.answer(), expected)
            self.assertEqual(arithmetic_rule(operation, task), expected)
        with self.assertRaises(ValueError):
            arithmetic_rule("external", task)

    def test_training_and_evaluation_have_disjoint_operands(self):
        train, evaluation = tasks(1, "train", 200), tasks(1, "eval", 200)
        self.assertTrue(all(0 <= task.a <= 9 and 0 <= task.b <= 9 for task in train))
        self.assertTrue(all(10 <= task.a <= 29 and 10 <= task.b <= 29 for task in evaluation))
        self.assertFalse({task.task_id for task in train} & {task.task_id for task in evaluation})

    def test_every_role_receives_self_and_both_peer_feedback(self):
        frame = ArithmeticTeam(0, Config(), True).episode(Task("fixture", 7, 4, "add"), True)
        self.assertEqual(len(frame["feedback"]), 9)
        self.assertEqual(len({event["event_id"] for event in frame["feedback"]}), 9)
        for role in ("generator", "refiner", "selector"):
            inbox = [event for event in frame["feedback"] if event["recipient"] == role]
            self.assertEqual({event["kind"] for event in inbox}, {"self", "peer"})
            self.assertTrue(all(event["received"] for event in inbox))

    def test_no_communication_keeps_only_self_feedback(self):
        frame = ArithmeticTeam(0, Config(), False).episode(Task("fixture", 7, 4, "add"), True)
        self.assertEqual(len(frame["feedback"]), 3)
        self.assertTrue(all(event["kind"] == "self" for event in frame["feedback"]))
        for update in frame["updates"]:
            self.assertEqual(update["reward"], update["self_score"])
            self.assertEqual(update["peer_score"], 0)

    def test_learning_updates_all_three_values_from_recorded_rewards(self):
        config = Config(alpha=0.25)
        frame = ArithmeticTeam(0, config, True).episode(Task("fixture", 7, 4, "add"), True)
        for update in frame["updates"]:
            expected = update["old_value"] + config.alpha * (update["reward"] - update["old_value"])
            self.assertAlmostEqual(update["new_value"], expected)
            self.assertEqual(update["version_after"], update["version_before"] + 1)

    def test_frozen_feedback_does_not_change_learning_state(self):
        team = ArithmeticTeam(0, Config(), True)
        for task in tasks(0, "train", 20):
            team.episode(task, True)
        before = {role: (policy.digest(), policy.version) for role, policy in team.policies.items()}
        for task in tasks(0, "eval", 20):
            frame = team.episode(task, False)
            self.assertEqual(len(frame["feedback"]), 9)
        self.assertEqual(before, {role: (policy.digest(), policy.version) for role, policy in team.policies.items()})

    def test_next_episode_starts_after_every_previous_update(self):
        team = ArithmeticTeam(0, Config(), True)
        for index in range(3):
            frame = team.episode(Task(f"fixture-{index}", 7, 4, "add"), True)
            self.assertTrue(all(update["version_before"] == index for update in frame["updates"]))
            self.assertTrue(all(update["version_after"] == index + 1 for update in frame["updates"]))

    def test_invalid_rewards_are_rejected(self):
        policy = Policy(("add",), 0, Config())
        for reward in (math.nan, math.inf, -1, 2):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                policy.learn("add", "add", reward)

    def test_positive_feedback_changes_values_and_action_preferences(self):
        policy = Policy(("add", "subtract"), 0, Config(epsilon=0))
        before = policy.digest()
        policy.learn("add", "add", 1.0)
        self.assertNotEqual(before, policy.digest())
        self.assertGreater(policy.value("add", "add"), policy.value("add", "subtract"))
        self.assertEqual({policy.choose("add") for _ in range(20)}, {"add"})

    def test_counts_use_matched_evaluation_tasks_and_frozen_policies(self):
        result = run_matrix(Config(train_episodes=20, eval_episodes=10, seeds=2))
        self.assertEqual(result["episodes"], 240)
        self.assertEqual(result["llm_calls"], 0)
        for seed in range(2):
            rows = [row for row in result["runs"] if row["seed"] == seed]
            self.assertEqual(len({row["eval_task_sha256"] for row in rows}), 1)
            for row in rows:
                self.assertTrue(row["frozen_eval_verified"])
                self.assertEqual(set(row["policy_updates"].values()), {20 if row["learning"] else 0})

    def test_repeatability_excludes_wall_clock(self):
        config = Config(train_episodes=20, eval_episodes=10, seeds=2)
        first, second = run_matrix(config), run_matrix(config)
        self.assertEqual(first["runs"], second["runs"])
        self.assertEqual(first["sampled_traces"], second["sampled_traces"])

    def test_sampled_feedback_ids_are_unique_across_interventions(self):
        result = run_matrix(Config(train_episodes=20, eval_episodes=10, seeds=2))
        ids = [event["event_id"] for frame in result["sampled_traces"] for event in frame["feedback"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_config_rejects_unbounded_or_nonfinite_runs(self):
        for config in (Config(seeds=21), Config(wall_seconds=math.nan),
                       Config(train_episodes=2000, eval_episodes=2000, seeds=20),
                       Config(epsilon=2), Config(trace_episodes=100)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_runtime_fuse_has_no_retry(self):
        guard = Guard(Config())
        guard.deadline = 0
        with self.assertRaises(FuseTripped):
            guard.tick()
        self.assertEqual(guard.episodes, 0)

    def test_episode_fuse_has_no_budget_overshoot(self):
        guard = Guard(Config())
        guard.limit = 1
        guard.tick()
        with self.assertRaises(FuseTripped):
            guard.tick()
        self.assertEqual(guard.episodes, 1)

    def test_cli_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(main(["--output", temporary, "--seeds", "1"]), 2)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_records_fuse_failure_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-run"
            with patch("benign_orca.benchmark.run_matrix", side_effect=FuseTripped("fixture fuse")) as runner:
                self.assertEqual(main(["--output", str(output), "--seeds", "1"]), 2)
                runner.assert_called_once()
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "aborted")
            self.assertEqual(failure["automatic_retries"], 0)
            self.assertFalse((output / "summary.json").exists())

    def test_source_has_no_model_or_attack_imports(self):
        source = Path(benchmark.__file__)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.split(".")[0] in
                             {"co_agent_fuzz", "PromptFuzz", "openai", "requests", "torch", "transformers"}
                             for name in modules))


if __name__ == "__main__":
    unittest.main()
