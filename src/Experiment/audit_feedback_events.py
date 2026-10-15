"""Read-only callback-delivery audit. No model calls or learning-state changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUBSCRIBERS = (
    "StrategyAgent.on_critic_feedback",
    "MutatorAgent.on_critic_feedback",
    "ArtifactStore.event_handler",
)
LIMITATION = (
    "Receipts verify callback completion and recorded ordering only. They do not "
    "verify learning-state changes, reward correctness, causal benefit or model performance."
)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def audit_events(events: list[dict], receipts: list[dict] | None = None,
                 manifest: dict | None = None,
                 required_subscribers: Iterable[str] = DEFAULT_SUBSCRIBERS) -> dict:
    """Do not infer successful delivery from publication or historical counters."""
    errors, gaps = [], []

    def issue(destination, code, iteration=None, line=None, subscriber=None):
        item = {"code": code}
        if iteration is not None:
            item["iteration"] = iteration
        if line is not None:
            item["line"] = line
        if subscriber is not None:
            item["subscriber"] = subscriber
        destination.append(item)

    if not events:
        issue(gaps, "empty_event_trace")
    required = tuple(required_subscribers)
    if not required or len(required) != len(set(required)):
        raise ValueError("Required subscribers must be nonempty and unique")
    if manifest is None or manifest.get("audit_schema_version") != 1:
        issue(gaps, "manifest_missing_or_unsupported")
    run_id = (manifest or {}).get("run_id")
    if not isinstance(run_id, str) or not run_id:
        issue(gaps, "run_identity_unavailable")
        run_id = None
    if receipts is None:
        issue(gaps, "delivery_trace_unavailable")

    by_iteration = defaultdict(list)
    feedback_by_id = {}
    event_ids = set()
    modern = True
    previous_clock = -1
    finished = []
    previous_cycle_iteration = 0
    for line, event in enumerate(events, 1):
        iteration = event.get("iteration")
        if not _integer(iteration):
            issue(errors, "invalid_iteration", line=line)
            continue
        if not isinstance(event.get("payload"), dict):
            issue(errors, "invalid_payload_shape", iteration, line)
            continue
        if not isinstance(event.get("type"), str) or not isinstance(event.get("sender"), str):
            issue(errors, "invalid_event_labels", iteration, line)
            continue
        if event["type"] in {"PHASE_STARTED", "CRITIC_FEEDBACK", "Q_UPDATE"}:
            if iteration < previous_cycle_iteration:
                issue(errors, "cycle_iteration_moved_backwards", iteration, line)
            previous_cycle_iteration = iteration
        by_iteration[iteration].append((line, event))
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            modern = False
        else:
            if event_id in event_ids:
                issue(errors, "duplicate_event_id", iteration, line)
            event_ids.add(event_id)
        sequence = event.get("sequence")
        clock = event.get("recorded_monotonic_ns")
        if sequence is None or clock is None or event.get("run_id") is None:
            modern = False
        else:
            if not _integer(sequence) or sequence != line:
                issue(errors, "noncontiguous_event_sequence", iteration, line)
            if not _integer(clock) or clock < previous_clock:
                issue(errors, "nonmonotonic_event_clock", iteration, line)
            else:
                previous_clock = clock
            if run_id is not None and event.get("run_id") != run_id:
                issue(errors, "event_run_mismatch", iteration, line)
        if event.get("type") == "CRITIC_FEEDBACK" and isinstance(event_id, str) and event_id:
            feedback_by_id[event_id] = (line, event)
        if event.get("type") in {"CRITIC_FEEDBACK", "Q_UPDATE"} and iteration == 0:
            issue(errors, "feedback_cycle_at_iteration_zero", iteration, line)
        if event.get("type") == "RUN_FINISHED":
            finished.append((line, event))
    if not modern:
        issue(gaps, "legacy_or_partial_event_metadata")
    if len(finished) != 1:
        issue(gaps if not finished else errors, "run_finish_marker_count")
    interrupted = any(str(event["payload"].get("stop_reason", "")).startswith("error:")
                      for _, event in finished)
    if interrupted:
        issue(gaps, "run_reported_interruption")
    if any(not isinstance(event["payload"].get("stop_reason"), str)
           or event["payload"].get("stop_reason") in ("", "running") for _, event in finished):
        issue(gaps, "run_completion_reason_unavailable")
    complete_run = len(finished) == 1 and not interrupted
    if finished:
        finish_line = finished[0][0]
        for line, event in enumerate(events, 1):
            if line > finish_line and event.get("type") in (
                    "CRITIC_FEEDBACK", "Q_UPDATE", "PHASE_STARTED"):
                issue(errors, "cycle_event_after_run_finished", event.get("iteration"), line)

    deliveries = defaultdict(list)
    dispatch_events = {}
    for line, receipt in enumerate(receipts or [], 1):
        event_id = receipt.get("event_id")
        match = feedback_by_id.get(event_id) if isinstance(event_id, str) else None
        if match is None:
            issue(errors, "orphan_delivery_receipt", line=line)
            continue
        _, event = match
        iteration = event["iteration"]
        if (not _integer(receipt.get("iteration"))
                or receipt.get("type") != event["type"]
                or receipt.get("iteration") != iteration
                or receipt.get("sender") != event.get("sender")
                or receipt.get("run_id") != event.get("run_id")):
            issue(errors, "receipt_correlation_mismatch", iteration, line)
        subscriber = receipt.get("subscriber")
        if not isinstance(subscriber, str) or not subscriber:
            issue(errors, "invalid_subscriber", iteration, line)
            continue
        if receipt.get("status") != "ok" or receipt.get("error_type") is not None:
            issue(errors, "handler_not_successful", iteration, line, subscriber)
        start, end = receipt.get("started_monotonic_ns"), receipt.get("finished_monotonic_ns")
        if not _integer(start) or not _integer(end) or end < start:
            issue(errors, "invalid_receipt_clock", iteration, line, subscriber)
        index, dispatch = receipt.get("subscriber_index"), receipt.get("dispatch_sequence")
        if not _integer(index) or not _integer(dispatch) or dispatch == 0:
            issue(errors, "invalid_delivery_sequence", iteration, line)
            continue
        else:
            previous_id = dispatch_events.setdefault(dispatch, event_id)
            if previous_id != event_id:
                issue(errors, "dispatch_sequence_reused", iteration, line)
            if _integer(event.get("sequence")) and dispatch != event["sequence"]:
                issue(errors, "receipt_event_sequence_mismatch", iteration, line)
        deliveries[event_id].append(receipt)

    strategy_entries = sorted(
        ((iteration, line, event) for iteration, entries in by_iteration.items()
         for line, event in entries if event["type"] == "PHASE_STARTED"
         and event["payload"].get("phase") == "Strategy-Agent"),
        key=lambda item: (item[0], item[1]))
    strategy_iterations = [item[0] for item in strategy_entries]
    unique_iterations = sorted(set(strategy_iterations))
    if unique_iterations and unique_iterations != list(range(1, len(unique_iterations) + 1)):
        issue(gaps, "iteration_coverage_gap")
    cycles = []
    feedback_count = 0
    for iteration, entries in sorted(by_iteration.items()):
        if iteration == 0:
            continue
        feedback = [(line, event) for line, event in entries
                    if event.get("type") == "CRITIC_FEEDBACK"]
        updates = [(line, event) for line, event in entries if event.get("type") == "Q_UPDATE"]
        critic = [line for line, event in entries
                  if event.get("type") == "PHASE_STARTED"
                  and event["payload"].get("phase") == "Critic-Agent"]
        starts = [line for line, event in entries
                  if event.get("type") == "PHASE_STARTED"
                  and event["payload"].get("phase") == "Strategy-Agent"]
        attacks = [line for line, event in entries
                   if event.get("type") == "PHASE_STARTED"
                   and event["payload"].get("phase") == "Attack-Agent"]
        if not (feedback or updates or critic or starts):
            continue
        feedback_count += len(feedback)
        before = len(errors) + len(gaps)
        destination = errors if complete_run else gaps
        for name, markers in (("strategy", starts), ("attack", attacks), ("critic", critic)):
            if len(markers) != 1:
                issue(errors if len(markers) > 1 else gaps, f"{name}_phase_marker_count", iteration)
        if starts and attacks and min(starts) >= min(attacks):
            issue(errors, "attack_before_strategy_phase", iteration)
        if attacks and critic and min(attacks) >= min(critic):
            issue(errors, "critic_before_attack_phase", iteration)
        if len(feedback) != 1:
            issue(errors if len(feedback) > 1 else destination, "feedback_marker_count", iteration)
        if len(updates) != 1:
            issue(errors if len(updates) > 1 else destination, "update_marker_count", iteration)
        if critic and feedback and min(critic) > feedback[0][0]:
            issue(errors, "feedback_before_critic_phase", iteration)
        if starts and critic and min(starts) > min(critic):
            issue(errors, "critic_before_strategy_phase", iteration)
        if feedback and updates and updates[0][0] <= feedback[0][0]:
            issue(errors, "update_marker_before_feedback", iteration)
        next_index = bisect_right(strategy_iterations, iteration)
        next_start = (strategy_entries[next_index][1:]
                      if next_index < len(strategy_entries) else None)
        if next_start and (not updates or updates[0][0] >= next_start[0]):
            issue(errors, "next_iteration_before_update_marker", iteration)
        receipt_count = 0
        verified = False
        if len(feedback) == 1:
            _, event = feedback[0]
            event_id = event.get("event_id")
            found = deliveries.get(event_id, []) if isinstance(event_id, str) else []
            receipt_count = len(found)
            subscribers, indices, dispatches = set(), set(), set()
            for receipt in found:
                name, index = receipt.get("subscriber"), receipt.get("subscriber_index")
                if name in subscribers or index in indices:
                    issue(errors, "duplicate_subscriber_receipt", iteration, subscriber=name)
                subscribers.add(name)
                indices.add(index)
                dispatches.add(receipt.get("dispatch_sequence"))
                end = receipt.get("finished_monotonic_ns")
                for boundary in ([updates[0]] if updates else []) + ([next_start] if next_start else []):
                    clock = boundary[1].get("recorded_monotonic_ns")
                    if _integer(end) and _integer(clock) and end > clock:
                        issue(errors, "delivery_after_cycle_boundary", iteration, subscriber=name)
            if len(dispatches) > 1:
                issue(errors, "feedback_dispatched_multiple_times", iteration)
            found_indices = [item["subscriber_index"] for item in found]
            if any(left >= right for left, right in zip(found_indices, found_indices[1:])):
                issue(errors, "subscriber_receipts_out_of_order", iteration)
            elif found and found_indices != list(range(len(found))):
                issue(gaps, "subscriber_receipt_sequence_gap", iteration)
            for previous, current in zip(found, found[1:]):
                end, start = previous.get("finished_monotonic_ns"), current.get("started_monotonic_ns")
                if _integer(end) and _integer(start) and start < end:
                    issue(errors, "overlapping_handler_receipts", iteration)
            for name in required:
                if name not in subscribers:
                    issue(gaps, "required_handler_receipt_missing", iteration, subscriber=name)
            verified = (modern and run_id is not None and receipts is not None
                        and set(required).issubset(subscribers)
                        and len(errors) + len(gaps) == before)
        cycles.append({"iteration": iteration, "feedback_markers": len(feedback),
                       "update_markers": len(updates), "receipts": receipt_count,
                       "delivery_verified": verified})
    if not cycles or feedback_count == 0:
        issue(gaps, "no_feedback_cycles_to_verify")
    status = "fail" if errors else "incomplete" if gaps else "pass"
    # A global identity/schema/order error invalidates any per-cycle certification.
    if status != "pass":
        for cycle in cycles:
            cycle["delivery_verified"] = False
    return {
        "audit_schema_version": 1, "status": status, "run_id": run_id,
        "events": len(events), "feedback_events": feedback_count,
        "receipt_count": len(receipts or []), "required_subscribers": list(required),
        "cycles": cycles, "errors": errors, "evidence_gaps": gaps,
        "limitation": LIMITATION,
    }


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object in {path.name}, line {line}")
            rows.append(value)
    return rows


def audit_run(run_dir: Path) -> dict:
    paths = {"events": run_dir / "events.jsonl", "receipts": run_dir / "delivery_receipts.jsonl",
             "manifest": run_dir / "manifest.json"}
    events = read_jsonl(paths["events"])
    receipts = read_jsonl(paths["receipts"]) if paths["receipts"].is_file() else None
    manifest = None
    if paths["manifest"].is_file():
        with paths["manifest"].open(encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object")
    result = audit_events(events, receipts, manifest)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    result["sources"] = {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items() if path.is_file()
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="New report only; existing files are never overwritten")
    args = parser.parse_args()
    try:
        result = audit_run(args.run_dir)
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        print(serialized, end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Audit could not complete: {exc}\n")
    return {"pass": 0, "fail": 1, "incomplete": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
