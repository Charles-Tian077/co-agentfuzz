"""Bounded, read-only audit of fixed arithmetic logs; no experiment execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


ROLES = ("generator", "refiner", "selector")
OPERATIONS = ("add", "subtract", "multiply")
CONDITIONS = {"full": (True, True), "no_learning": (False, True),
              "no_communication": (True, False), "neither": (False, False)}
SCOPE = "isolated arithmetic mechanism diagnostic"
MAX_BYTES = 8_000_000


class AuditError(ValueError):
    """Evidence is invalid or incomplete; do not silently repair it."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def integer(value, minimum, maximum, name):
    require(type(value) is int and minimum <= value <= maximum, f"Invalid {name}")
    return value


def number(value, name, minimum=0, maximum=1):
    require(type(value) in (int, float) and math.isfinite(value)
            and minimum <= value <= maximum, f"Invalid {name}")
    return value


def close(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), name)


def sha256(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "Invalid SHA256")


def fixed_tasks(seed, split, count):
    """Reconstruct the public arithmetic fixture, not a model or attack target."""
    rng = random.Random(seed * 1009 + (0 if split == "train" else 71))
    low, high = (0, 9) if split == "train" else (10, 29)
    return [{"task_id": f"{split}/{seed}/{index}", "a": rng.randint(low, high),
             "b": rng.randint(low, high), "operation": rng.choice(OPERATIONS)}
            for index in range(count)]


def calculate(operation, task):
    a, b = task["a"], task["b"]
    require(operation in OPERATIONS, "Unknown fixed arithmetic operation")
    return {"add": a + b, "subtract": a - b, "multiply": a * b}[operation]


def audit_records(summary, frames):
    """Validate identities, arithmetic labels, receipts, rewards and value updates."""
    require(type(summary["schema_version"]) is int and summary["schema_version"] == 1
            and summary["status"] == "complete"
            and summary["scope"] == SCOPE, "Not a complete benign arithmetic run")
    UUID(summary["run_id"])
    sha256(summary["source_sha256"])
    config = summary["config"]
    train = integer(config["train_episodes"], 1, 2000, "train episodes")
    evaluation = integer(config["eval_episodes"], 1, 2000, "eval episodes")
    seeds = integer(config["seeds"], 1, 20, "seeds")
    sample = integer(config["trace_episodes"], 0, 5, "sample size")
    alpha = number(config["alpha"], "alpha", 0.001)
    number(config["epsilon"], "epsilon")
    number(config["wall_seconds"], "wall budget", 0.01, 120)
    limit = integer(config["episode_limit"], 1, 50000, "episode limit")
    total = 4 * seeds * (train + evaluation)
    require(total <= limit and type(summary["episodes"]) is int
            and summary["episodes"] == total, "Episode accounting mismatch")
    number(summary["wall_seconds"], "elapsed seconds", 0, 86400)
    # Missing costs are errors, not zeros. Zero is only valid in this no-LLM scope.
    integer(summary["llm_calls"], 0, 0, "LLM calls")
    integer(summary["llm_tokens"], 0, 0, "LLM tokens")
    require(len(summary["runs"]) == 4 * seeds, "Incomplete trial matrix")
    require(set(summary["summary_accuracy"]) == set(CONDITIONS), "Condition mismatch")
    trials, fixtures = {}, {}
    for seed in range(seeds):
        for split, count in (("train", train), ("eval", evaluation)):
            fixtures[seed, split] = fixed_tasks(seed, split, count)
    for row in summary["runs"]:
        seed = integer(row["seed"], 0, seeds - 1, "trial seed")
        condition = row["condition"]
        require(condition in CONDITIONS, "Unknown condition")
        key = seed, condition
        require(key not in trials, "Duplicate trial")
        learning, communication = CONDITIONS[condition]
        require(row["learning"] is learning and row["communication"] is communication,
                "Trial intervention mismatch")
        integer(row["train_successes"], 0, train, "train successes")
        integer(row["eval_successes"], 0, evaluation, "eval successes")
        require(type(row["eval_tasks"]) is int and row["eval_tasks"] == evaluation,
                "Eval denominator mismatch")
        close(row["eval_accuracy"], row["eval_successes"] / evaluation, "Trial accuracy")
        digest = hashlib.sha256(json.dumps(fixtures[seed, "eval"], sort_keys=True)
                                .encode("utf-8")).hexdigest()
        require(row["eval_task_sha256"] == digest, "Eval task digest mismatch")
        require(type(row["feedback_receipts"]) is int and row["feedback_receipts"]
                == (9 if communication else 3) * (train + evaluation), "Receipt total")
        require(set(row["policy_updates"]) == set(ROLES), "Missing policy role")
        for value in row["policy_updates"].values():
            require(type(value) is int and value == (train if learning else 0),
                    "Policy update count mismatch")
        require(row["frozen_eval_verified"] is True, "Frozen evaluation not reported")
        require(set(row["policy_sha256"]) == set(ROLES), "Missing policy digest")
        for value in row["policy_sha256"].values():
            sha256(value)
        trials[key] = row
    aggregates = []
    for condition in CONDITIONS:
        rows = [row for (_, name), row in trials.items() if name == condition]
        successes = sum(row["eval_successes"] for row in rows)
        accuracy = successes / (seeds * evaluation)
        close(summary["summary_accuracy"][condition], accuracy, "Summary accuracy")
        aggregates.append({"condition": condition, "seeds": seeds,
                           "eval_successes": successes, "eval_tasks": seeds * evaluation,
                           "accuracy": accuracy, "feedback_receipts": sum(
                               row["feedback_receipts"] for row in rows)})

    require(len(frames) == 4 * seeds * (min(sample, train) + min(sample, evaluation)),
            "Sample coverage mismatch")
    seen_frames, seen_events, last_sample = set(), set(), {}
    receipts = Counter()
    changed, learning_calls = Counter(), Counter()
    for frame in frames:
        seed, condition = frame["seed"], frame["condition"]
        require(type(seed) is int and (seed, condition) in trials, "Orphan sample")
        trial_id = f"seed-{seed}/{condition}"
        require(frame["run_id"] == summary["run_id"] and frame["trial_id"] == trial_id,
                "Sample run/trial identity mismatch")
        parts = frame["task_id"].split("/")
        require(len(parts) == 3 and parts[0] in ("train", "eval")
                and parts[1] == str(seed) and parts[2].isdigit(), "Invalid sampled task")
        split, index = parts[0], int(parts[2])
        count = train if split == "train" else evaluation
        require(index < min(sample, count), "Unexpected sample index")
        task = fixtures[seed, split][index]
        require(frame["task_id"] == task["task_id"], "Noncanonical task identity")
        key = seed, condition, split, index
        require(key not in seen_frames, "Duplicate sample")
        seen_frames.add(key)
        learning, communication = CONDITIONS[condition]
        learn = learning and split == "train"
        require(frame["learning"] is learn and frame["communication"] is communication
                and type(frame["success"]) is bool, "Sample intervention mismatch")
        pairs = {(sender, recipient) for sender in ROLES for recipient in ROLES
                 if communication or sender == recipient}
        events = frame["feedback"]
        require(len(events) == len(pairs), "Incomplete feedback barrier")
        scores, actual_pairs = {}, set()
        for sequence, event in enumerate(events, 1):
            sender, recipient = event["sender"], event["recipient"]
            require((sender, recipient) in pairs and (sender, recipient) not in actual_pairs,
                    "Duplicate or unexpected feedback route")
            actual_pairs.add((sender, recipient))
            event_id = f"{trial_id}/{task['task_id']}/{sender}/{recipient}"
            require(event["event_id"] == event_id and event_id not in seen_events,
                    "Feedback identity mismatch or duplicate")
            seen_events.add(event_id)
            kind = "self" if sender == recipient else "peer"
            require(event["task_id"] == task["task_id"] and event["kind"] == kind
                    and event["received"] is True and type(event["receipt_sequence"]) is int
                    and event["receipt_sequence"] == sequence, "Receipt mismatch")
            score = number(event["verified_score"], "verified score")
            require(score in (0, 1) and scores.get(sender, score) == score,
                    "Inconsistent sender score")
            scores[sender] = score
            receipts[kind] += 1
        updates = frame["updates"]
        require(len(updates) == 3 and [update["role"] for update in updates] == list(ROLES),
                "Incomplete or reordered role updates")
        gen, ref, sel = updates
        original = calculate(gen["action"], task)
        require(ref["action"] in ("keep",) + OPERATIONS and sel["action"] in ROLES[:2],
                "Unknown arithmetic action")
        revised = original if ref["action"] == "keep" else calculate(ref["action"], task)
        selected = original if sel["action"] == "generator" else revised
        expected = calculate(task["operation"], task)
        truth = {"generator": float(original == expected), "refiner": float(revised == expected),
                 "selector": float(selected == expected)}
        require(scores == truth and frame["success"] == bool(truth["selector"]),
                "Arithmetic feedback label mismatch")
        ancestry = sel["action"] == "generator" or ref["action"] == "keep"
        peer = {"generator": truth["selector"] * ancestry,
                "refiner": truth["selector"] * (sel["action"] == "refiner"),
                "selector": truth[sel["action"]]} if communication else dict.fromkeys(ROLES, 0)
        relation = "tie" if gen["old_value"] == ref["old_value"] else (
            "generator-higher" if gen["old_value"] > ref["old_value"] else "refiner-higher")
        states = {"generator": task["operation"], "refiner": task["operation"] + (
            ("/peer-high" if gen["old_value"] >= 0.5 else "/peer-low") if communication else ""),
            "selector": task["operation"] + "/" + (relation if communication else "unobserved")}
        previous = last_sample.get((seed, condition, split))
        require(index == (0 if previous is None else previous[0] + 1), "Sample ordering gap")
        for update in updates:
            role = update["role"]
            require(update["state"] == states[role], "Unexpected arithmetic state")
            close(update["self_score"], truth[role], "Self feedback mismatch")
            close(update["peer_score"], peer[role], "Peer attribution mismatch")
            reward = 0.8 * truth[role] + 0.2 * peer[role] if communication else truth[role]
            close(update["reward"], reward, "Reward equation mismatch")
            old = number(update["old_value"], "old value")
            new = old + alpha * (reward - old) if learn else old
            close(update["new_value"], new, "Value update equation mismatch")
            version = index if learn else (train if learning and split == "eval" else 0)
            require(type(update["version_before"]) is int and update["version_before"] == version
                    and type(update["version_after"]) is int
                    and update["version_after"] == version + int(learn), "Update barrier mismatch")
            if previous is not None:
                prev = previous[1][role]
                require(prev["version_after"] == update["version_before"], "Next-episode barrier")
                if (prev["state"], prev["action"]) == (update["state"], update["action"]):
                    close(old, prev["new_value"], "Consecutive value continuity")
            if learn:
                learning_calls[role] += 1
                changed[role] += not math.isclose(old, update["new_value"], rel_tol=1e-12, abs_tol=1e-12)
        last_sample[seed, condition, split] = index, {u["role"]: u for u in updates}
    return {"status": "passed", "scope": SCOPE, "run_id": summary["run_id"],
            "trials": len(trials), "episodes": total, "aggregates": aggregates,
            "sampled_episodes_checked": len(frames), "sampled_receipts": dict(receipts),
            "sampled_learning_calls": dict(learning_calls),
            "sampled_nonzero_value_changes": dict(changed),
            "llm_calls": 0, "llm_tokens": 0,
            "limitations": ["Samples, not full independent trace certification.",
                            "Frozen final policy digests are reported, not recomputed from full traces.",
                            "Arithmetic contextual bandits, not LLM training or attack performance.",
                            "Zero tokens means no LLM; token-saving percentage is undefined."]}


def read_run(directory):
    paths = [directory / "summary.json", directory / "sampled_traces.jsonl"]
    raw = []
    for path in paths:
        require(path.stat().st_size <= MAX_BYTES, "Input exceeds audit size limit")
        raw.append(path.read_bytes())
    summary = json.loads(raw[0].decode("utf-8"))
    lines = raw[1].decode("utf-8").splitlines()
    require(len(lines) <= 800, "Too many sampled frames")
    frames = [json.loads(line) for line in lines]
    return summary, frames, {path.name: hashlib.sha256(data).hexdigest()
                            for path, data in zip(paths, raw)}


def audit_run(directory, companion=None):
    summary, frames, hashes = read_run(Path(directory))
    result = audit_records(summary, frames)
    result["input_sha256"] = hashes
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    if companion is not None:
        other, other_frames, other_hashes = read_run(Path(companion))
        audit_records(other, other_frames)
        for field in ("scope", "config", "episodes", "source_sha256", "summary_accuracy",
                      "llm_calls", "llm_tokens"):
            require(summary[field] == other[field], f"Companion {field} mismatch")
        key = lambda row: (row["seed"], row["condition"])
        require(sorted(summary["runs"], key=key) == sorted(other["runs"], key=key),
                "Companion trial results mismatch")
        # Run identities differ intentionally; all other sampled content must match.
        strip_id = lambda trace: {k: v for k, v in trace.items() if k != "run_id"}
        require([strip_id(f) for f in frames] == [strip_id(f) for f in other_frames],
                "Companion sample mismatch")
        result["companion_sha256"] = other_hashes
        result["paired_local_results_match"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--companion", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file only; never replaces evidence")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            require(not args.output.exists(), "Refusing to overwrite an existing output")
        result = audit_run(args.directory, args.companion)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, IndexError, AttributeError) as error:
        print(f"Benign audit stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
