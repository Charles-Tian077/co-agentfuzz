"""Three contextual bandits on fixed integer arithmetic, using stdlib only.

This module deliberately does not import the existing Co-AgentFuzz implementation.
It accepts no prompts, external datasets, model clients or arbitrary operators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from uuid import uuid4
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROLES = ("generator", "refiner", "selector")
OPERATIONS = ("add", "subtract", "multiply")
CONDITIONS = {
    "full": (True, True),
    "no_learning": (False, True),
    "no_communication": (True, False),
    "neither": (False, False),
}


class FuseTripped(RuntimeError):
    """A bounded run stopped; the caller must not retry automatically."""


@dataclass(frozen=True)
class Config:
    train_episodes: int = 400
    eval_episodes: int = 200
    seeds: int = 8
    epsilon: float = 0.15
    alpha: float = 0.25
    wall_seconds: float = 30.0
    episode_limit: int = 50000
    trace_episodes: int = 2

    def validate(self):
        for name, minimum, maximum in (
            ("train_episodes", 1, 2000), ("eval_episodes", 1, 2000),
            ("seeds", 1, 20), ("episode_limit", 1, 50000),
            ("trace_episodes", 0, 5),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        for name, minimum, maximum in (
            ("epsilon", 0.0, 1.0), ("alpha", 0.001, 1.0),
            ("wall_seconds", 0.01, 120.0),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
        total = 4 * self.seeds * (self.train_episodes + self.eval_episodes)
        if total > self.episode_limit:
            raise ValueError("Planned episodes exceed the hard episode limit")


class Guard:
    def __init__(self, config: Config):
        self.limit = config.episode_limit
        self.deadline = time.monotonic() + config.wall_seconds
        self.episodes = 0

    def tick(self):
        if self.episodes >= self.limit:
            raise FuseTripped("episode limit reached")
        if time.monotonic() >= self.deadline:
            raise FuseTripped("wall-clock limit reached")
        self.episodes += 1


@dataclass(frozen=True)
class Task:
    task_id: str
    a: int
    b: int
    operation: str

    def answer(self) -> int:
        if self.operation == "add":
            return self.a + self.b
        if self.operation == "subtract":
            return self.a - self.b
        if self.operation == "multiply":
            return self.a * self.b
        raise ValueError("Only fixed arithmetic operations are supported")


def arithmetic_rule(operation: str, task: Task) -> int:
    if operation == "add":
        return sum((task.a, task.b))
    if operation == "subtract":
        return task.a + (-task.b)
    if operation == "multiply":
        return task.a * task.b
    raise ValueError("Unknown arithmetic rule")


def tasks(seed: int, split: str, count: int) -> list[Task]:
    if split not in ("train", "eval"):
        raise ValueError("Unknown split")
    rng = random.Random(seed * 1009 + (0 if split == "train" else 71))
    low, high = (0, 9) if split == "train" else (10, 29)
    return [Task(f"{split}/{seed}/{index}", rng.randint(low, high),
                 rng.randint(low, high), rng.choice(OPERATIONS)) for index in range(count)]


class Policy:
    """One-step contextual value learning, not LLM-weight training or PPO."""

    def __init__(self, actions: tuple[str, ...], seed: int, config: Config):
        self.actions, self.config = actions, config
        self.rng = random.Random(seed)
        self.values: dict[str, dict[str, float]] = {}
        self.version = 0

    def value(self, state: str, action: str) -> float:
        return self.values.get(state, {}).get(action, 0.0)

    def choose(self, state: str) -> str:
        # Always consume both draws, so intervention pairs use matched RNG streams.
        exploration, tie = self.rng.random(), self.rng.random()
        scores = [self.value(state, action) for action in self.actions]
        options = self.actions if exploration < self.config.epsilon else tuple(
            action for action, score in zip(self.actions, scores) if score == max(scores))
        return options[min(int(tie * len(options)), len(options) - 1)]

    def learn(self, state: str, action: str, reward: float):
        if action not in self.actions or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Invalid arithmetic learning record")
        old = self.value(state, action)
        new = old + self.config.alpha * (reward - old)
        if not math.isfinite(new):
            raise FuseTripped("non-finite learning state")
        self.values.setdefault(state, {})[action] = new
        self.version += 1

    def digest(self) -> str:
        serialized = json.dumps(self.values, sort_keys=True, allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ArithmeticTeam:
    def __init__(self, seed: int, config: Config, communication: bool,
                 trial_id: str = "fixture"):
        self.communication = communication
        self.trial_id = trial_id
        self.policies = {
            "generator": Policy(OPERATIONS, seed * 31 + 1, config),
            "refiner": Policy(("keep",) + OPERATIONS, seed * 31 + 2, config),
            "selector": Policy(("generator", "refiner"), seed * 31 + 3, config),
        }

    def episode(self, task: Task, learn: bool) -> dict:
        generator, refiner, selector = (self.policies[role] for role in ROLES)
        before = {role: policy.version for role, policy in self.policies.items()}
        generator_state = task.operation
        generator_action = generator.choose(generator_state)
        original = arithmetic_rule(generator_action, task)
        generator_belief = generator.value(generator_state, generator_action)
        # The proposal is an essential workflow input even in no-communication.
        # Only optional value/confidence and peer-feedback messages are removed.
        refiner_state = task.operation
        if self.communication:
            refiner_state += "/peer-high" if generator_belief >= 0.5 else "/peer-low"
        refiner_action = refiner.choose(refiner_state)
        revised = original if refiner_action == "keep" else arithmetic_rule(refiner_action, task)
        refiner_belief = refiner.value(refiner_state, refiner_action)
        relation = "tie"
        if generator_belief > refiner_belief:
            relation = "generator-higher"
        elif refiner_belief > generator_belief:
            relation = "refiner-higher"
        selector_state = task.operation + "/" + (relation if self.communication else "unobserved")
        selected = selector.choose(selector_state)
        chosen = original if selected == "generator" else revised

        # Labels are computed only after selection, never included in its input.
        expected = task.answer()
        local = {"generator": float(original == expected),
                 "refiner": float(revised == expected),
                 "selector": float(chosen == expected)}
        feedback = []
        for sender in ROLES:
            for recipient in ROLES:
                if sender != recipient and not self.communication:
                    continue
                feedback.append({"event_id": f"{self.trial_id}/{task.task_id}/{sender}/{recipient}",
                                 "task_id": task.task_id, "sender": sender,
                                 "recipient": recipient, "kind": "self" if sender == recipient else "peer",
                                 "verified_score": local[sender], "received": False})
        inbox = {role: {} for role in ROLES}
        for sequence, event in enumerate(feedback, 1):
            recipient, sender = event["recipient"], event["sender"]
            if sender in inbox[recipient]:
                raise FuseTripped("duplicate arithmetic feedback")
            inbox[recipient][sender] = event["verified_score"]
            event.update(received=True, receipt_sequence=sequence)
        expected_senders = set(ROLES) if self.communication else None
        for role in ROLES:
            if set(inbox[role]) != (expected_senders if self.communication else {role}):
                raise FuseTripped("incomplete arithmetic feedback barrier")

        # Fixed illustrative attribution, not a tuned or novel reward algorithm.
        # Generation contributes through a selected proposal or a copied revision.
        ancestry = selected == "generator" or (selected == "refiner" and refiner_action == "keep")
        peer = {"generator": inbox["generator"].get("selector", 0.0) * ancestry,
                "refiner": inbox["refiner"].get("selector", 0.0) * (selected == "refiner"),
                "selector": inbox["selector"].get(selected, 0.0)}
        rewards = {role: 0.8 * inbox[role][role] + 0.2 * peer[role]
                   if self.communication else inbox[role][role] for role in ROLES}
        states = dict(zip(ROLES, (generator_state, refiner_state, selector_state)))
        actions = dict(zip(ROLES, (generator_action, refiner_action, selected)))
        updates = []
        for role in ROLES:
            policy = self.policies[role]
            old = policy.value(states[role], actions[role])
            if learn:
                policy.learn(states[role], actions[role], rewards[role])
            updates.append({"role": role, "state": states[role], "action": actions[role],
                            "self_score": inbox[role][role], "peer_score": peer[role],
                            "reward": rewards[role], "old_value": old,
                            "new_value": policy.value(states[role], actions[role]),
                            "version_before": before[role], "version_after": policy.version})
        # Returning only after all three updates is the next-episode barrier.
        return {"trial_id": self.trial_id, "task_id": task.task_id, "learning": learn,
                "communication": self.communication, "success": bool(local["selector"]),
                "feedback": feedback, "updates": updates}


def run_matrix(config: Config, verbose: bool = False) -> dict:
    config.validate()
    guard, rows, sampled_traces = Guard(config), [], []
    started = time.monotonic()
    for seed in range(config.seeds):
        training = tasks(seed, "train", config.train_episodes)
        evaluation = tasks(seed, "eval", config.eval_episodes)
        digest = hashlib.sha256(json.dumps([asdict(task) for task in evaluation], sort_keys=True).encode()).hexdigest()
        for condition, (learning, communication) in CONDITIONS.items():
            team = ArithmeticTeam(seed, config, communication, f"seed-{seed}/{condition}")
            feedback_count = 0
            train_success = 0
            for index, task in enumerate(training):
                guard.tick()
                frame = team.episode(task, learning)
                train_success += frame["success"]
                feedback_count += len(frame["feedback"])
                if index < config.trace_episodes:
                    sampled_traces.append({"seed": seed, "condition": condition, **frame})
            trained = {role: policy.digest() for role, policy in team.policies.items()}
            eval_success = 0
            for index, task in enumerate(evaluation):
                guard.tick()
                frame = team.episode(task, False)
                eval_success += frame["success"]
                feedback_count += len(frame["feedback"])
                if index < config.trace_episodes:
                    sampled_traces.append({"seed": seed, "condition": condition, **frame})
            if trained != {role: policy.digest() for role, policy in team.policies.items()}:
                raise FuseTripped("learning state changed during frozen evaluation")
            row = {"seed": seed, "condition": condition, "learning": learning,
                   "communication": communication, "train_successes": train_success,
                   "eval_successes": eval_success, "eval_tasks": len(evaluation),
                   "eval_accuracy": eval_success / len(evaluation),
                   "eval_task_sha256": digest, "feedback_receipts": feedback_count,
                   "policy_updates": {role: policy.version for role, policy in team.policies.items()},
                   "frozen_eval_verified": True, "policy_sha256": trained}
            rows.append(row)
            if verbose:
                print(f"seed={seed} condition={condition} eval={eval_success}/{len(evaluation)} frozen=yes", flush=True)
    summary = {condition: sum(row["eval_successes"] for row in rows if row["condition"] == condition)
               / (config.seeds * config.eval_episodes) for condition in CONDITIONS}
    return {"schema_version": 1, "status": "complete", "scope": "isolated arithmetic mechanism diagnostic",
            "generated_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
            "episodes": guard.episodes, "wall_seconds": time.monotonic() - started,
            "llm_calls": 0, "llm_tokens": 0, "summary_accuracy": summary,
            "runs": rows, "sampled_traces": sampled_traces,
            "limitations": ["Contextual bandits, not neural or LLM policy training.",
                            "Communication removes confidence messages and peer rewards jointly.",
                            "No significance, token-efficiency or Co-AgentFuzz attack-performance claim.",
                            "Toy arithmetic labels and exploration are not evidence of general collaboration."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--wall-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    config = Config(train_episodes=args.train_episodes, eval_episodes=args.eval_episodes,
                    seeds=args.seeds, wall_seconds=args.wall_seconds)
    created = False
    try:
        config.validate()
        args.output.mkdir(parents=True, exist_ok=False)
        created = True
        result = run_matrix(config, verbose=True)
        traces = result.pop("sampled_traces")
        result["run_id"] = str(uuid4())
        result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with (args.output / "sampled_traces.jsonl").open("x", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps({"run_id": result["run_id"], **trace},
                                        ensure_ascii=False, allow_nan=False) + "\n")
        with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"status": result["status"], "summary_accuracy": result["summary_accuracy"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, FuseTripped) as error:
        if created:
            try:
                with (args.output / "failure.json").open("x", encoding="utf-8") as handle:
                    json.dump({"status": "aborted", "reason": str(error),
                               "automatic_retries": 0, "config": asdict(config)},
                              handle, ensure_ascii=False, indent=2)
            except OSError:
                pass
        print(f"Benign run stopped without retry: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
