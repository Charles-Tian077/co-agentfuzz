"""Stop a benign pilot when its completed smoke yielded no visible answers.

This is a resource guard, not a scorer or a change to the three-role study.
"""

import argparse
import json
from pathlib import Path


def verify_smoke(path):
    root = Path(path)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or audit.get("status") != "passed":
        raise ValueError("Smoke completion and audit are required")
    if summary.get("model_calls") != 72:
        raise ValueError("Unexpected smoke call count")
    rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    steps = [step for row in rows for step in row["steps"]]
    if len(steps) != 72:
        raise ValueError("Smoke ledger does not contain 72 model steps")
    visible = sum(bool(step["text"].strip()) for step in steps)
    successes = sum(int(trial.get("successes", 0)) for trial in summary.get("trials", []))
    full_successes = sum(int(trial.get("successes", 0)) for trial in summary.get("trials", [])
                         if trial.get("condition") == "full")
    learning_trials = [trial for trial in summary.get("trials", [])
                       if trial.get("condition") in ("full", "no_communication")]
    q_changes = sum(int(trial.get("q_changes", 0)) for trial in learning_trials)
    result = {"episodes": len(rows), "model_calls": len(steps), "visible_responses": visible,
              "empty_responses": len(steps) - visible, "successes": successes,
              "full_successes": full_successes, "q_changes": q_changes,
              "pilot_allowed": visible > 0 and full_successes > 0 and q_changes > 0}
    print(json.dumps(result, sort_keys=True), flush=True)
    if visible == 0:
        raise SystemExit("Pilot fused: every smoke response is empty")
    if full_successes == 0:
        raise SystemExit("Pilot fused: full-condition smoke accuracy is zero")
    if q_changes == 0:
        raise SystemExit("Pilot fused: learning conditions produced no nonzero Q changes")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke")
    verify_smoke(parser.parse_args().smoke)
