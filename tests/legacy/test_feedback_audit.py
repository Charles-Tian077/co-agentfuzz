"""Neutral synthetic messages only; no coordinator runs, model calls or training."""

import asyncio
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import Experiment

from Experiment.audit_feedback_events import DEFAULT_SUBSCRIBERS, audit_events, audit_run, read_jsonl
from co_agent_fuzz.core.artifacts import ArtifactStore
from co_agent_fuzz.core.message_bus import Message, MessageBus, MessageType


def trace(iterations=2):
    events, receipts = [], []
    manifest = {"run_id": "fixture-run", "audit_schema_version": 1}

    def event(kind, iteration, payload):
        sequence = len(events) + 1
        item = {"type": kind, "iteration": iteration, "payload": payload,
                "sender": "fixture", "event_id": f"event-{sequence}",
                "run_id": manifest["run_id"], "sequence": sequence,
                "recorded_monotonic_ns": sequence * 100}
        events.append(item)
        return item

    for iteration in range(1, iterations + 1):
        for phase in ("Strategy-Agent", "Attack-Agent", "Critic-Agent"):
            event("PHASE_STARTED", iteration, {"phase": phase})
        feedback = event("CRITIC_FEEDBACK", iteration, {"document_id": f"doc-{iteration}"})
        for index, subscriber in enumerate(DEFAULT_SUBSCRIBERS):
            start = feedback["recorded_monotonic_ns"] - 50 + index * 20
            receipts.append({
                "event_id": feedback["event_id"], "type": feedback["type"],
                "sender": feedback["sender"], "iteration": iteration,
                "run_id": manifest["run_id"], "dispatch_sequence": feedback["sequence"],
                "subscriber_index": index, "subscriber": subscriber,
                "status": "ok", "error_type": None,
                "started_monotonic_ns": start, "finished_monotonic_ns": start + 10,
            })
        event("Q_UPDATE", iteration, {})  # A marker, not evidence of learning.
    event("RUN_FINISHED", iterations, {"stop_reason": "fixture_complete"})
    return events, receipts, manifest


class OfflineAuditTests(unittest.TestCase):
    def check_code(self, result, code, status="fail"):
        self.assertEqual(result["status"], status)
        self.assertIn(code, {item["code"] for item in result["errors"] + result["evidence_gaps"]})
        self.assertFalse(any(item["delivery_verified"] for item in result["cycles"]))

    def test_clean_trace_passes_delivery_only(self):
        result = audit_events(*trace())
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["feedback_events"], 2)
        self.assertTrue(all(item["delivery_verified"] for item in result["cycles"]))
        self.assertIn("do not verify learning", result["limitation"])
        self.assertNotIn("document_id", json.dumps(result))

    def test_duplicate_event_and_feedback_are_detected(self):
        events, receipts, manifest = trace()
        events.insert(4, copy.deepcopy(events[3]))
        self.check_code(audit_events(events, receipts, manifest), "duplicate_event_id")
        self.check_code(audit_events(events, receipts, manifest), "feedback_marker_count")

    def test_missing_delivery_trace_is_not_success(self):
        events, _, manifest = trace()
        self.check_code(audit_events(events, None, manifest), "delivery_trace_unavailable", "incomplete")

    def test_missing_one_receipt_is_incomplete(self):
        events, receipts, manifest = trace()
        receipts.pop(2)
        self.check_code(audit_events(events, receipts, manifest), "required_handler_receipt_missing", "incomplete")

    def test_missing_middle_receipt_is_incomplete_not_a_handler_failure(self):
        events, receipts, manifest = trace()
        receipts.pop(1)
        self.check_code(audit_events(events, receipts, manifest), "required_handler_receipt_missing", "incomplete")

    def test_callback_error_fails(self):
        events, receipts, manifest = trace()
        receipts[0].update(status="error", error_type="ValueError")
        self.check_code(audit_events(events, receipts, manifest), "handler_not_successful")

    def test_interrupted_callback_fails(self):
        events, receipts, manifest = trace()
        receipts[0]["status"] = "interrupted"
        self.check_code(audit_events(events, receipts, manifest), "handler_not_successful")

    def test_cross_iteration_or_run_receipts_fail(self):
        for key, value in (("iteration", 2), ("iteration", True), ("run_id", "other"), ("sender", "other")):
            with self.subTest(key=key, value=value):
                events, receipts, manifest = trace()
                receipts[0][key] = value
                self.check_code(audit_events(events, receipts, manifest), "receipt_correlation_mismatch")

    def test_orphan_receipt_fails(self):
        events, receipts, manifest = trace()
        receipts[0]["event_id"] = "nonexistent"
        self.check_code(audit_events(events, receipts, manifest), "orphan_delivery_receipt")

    def test_duplicate_subscriber_receipt_fails(self):
        events, receipts, manifest = trace()
        receipts.insert(1, copy.deepcopy(receipts[0]))
        self.check_code(audit_events(events, receipts, manifest), "duplicate_subscriber_receipt")

    def test_sequence_correlation_fails(self):
        events, receipts, manifest = trace()
        receipts[0]["dispatch_sequence"] = 99
        self.check_code(audit_events(events, receipts, manifest), "receipt_event_sequence_mismatch")

    def test_update_before_feedback_fails(self):
        events, receipts, manifest = trace()
        events[3], events[4] = events[4], events[3]
        self.check_code(audit_events(events, receipts, manifest), "update_marker_before_feedback")

    def test_next_iteration_before_previous_update_fails(self):
        events, receipts, manifest = trace()
        events[4], events[5] = events[5], events[4]
        self.check_code(audit_events(events, receipts, manifest), "next_iteration_before_update_marker")

    def test_late_delivery_fails(self):
        events, receipts, manifest = trace()
        receipts[0]["finished_monotonic_ns"] = 1000
        self.check_code(audit_events(events, receipts, manifest), "delivery_after_cycle_boundary")

    def test_handler_overlap_fails(self):
        events, receipts, manifest = trace()
        receipts[1]["started_monotonic_ns"] = receipts[0]["finished_monotonic_ns"] - 1
        self.check_code(audit_events(events, receipts, manifest), "overlapping_handler_receipts")

    def test_legacy_trace_does_not_certify_delivery(self):
        events, _, _ = trace()
        for event in events:
            for field in ("run_id", "event_id", "sequence", "recorded_monotonic_ns"):
                event.pop(field)
        self.check_code(audit_events(events), "legacy_or_partial_event_metadata", "incomplete")

    def test_incomplete_run_is_not_certified(self):
        events, receipts, manifest = trace()
        events.pop()
        self.check_code(audit_events(events, receipts, manifest), "run_finish_marker_count", "incomplete")

    def test_reported_run_error_is_not_certified(self):
        events, receipts, manifest = trace()
        events[-1]["payload"]["stop_reason"] = "error:InterruptedError"
        self.check_code(audit_events(events, receipts, manifest), "run_reported_interruption", "incomplete")

    def test_completed_run_missing_feedback_fails(self):
        events, _, manifest = trace(1)
        events.pop(3)
        self.check_code(audit_events(events, [], manifest), "feedback_marker_count")

    def test_empty_trace_is_incomplete(self):
        self.check_code(audit_events([], [], {"run_id": "fixture", "audit_schema_version": 1}),
                        "empty_event_trace", "incomplete")

    def test_invalid_metadata_does_not_crash(self):
        for field, value, code in (
                ("event_id", [], "legacy_or_partial_event_metadata"),
                ("iteration", True, "invalid_iteration"),
                ("type", [], "invalid_event_labels"),
                ("payload", [], "invalid_payload_shape"),
                ("recorded_monotonic_ns", -1, "nonmonotonic_event_clock")):
            with self.subTest(field=field):
                events, receipts, manifest = trace()
                events[0][field] = value
                result = audit_events(events, receipts, manifest)
                self.assertNotEqual(result["status"], "pass")
                self.assertIn(code, {item["code"] for item in result["errors"] + result["evidence_gaps"]})

    def test_invalid_receipt_metadata_does_not_crash(self):
        for field, value in (("subscriber", []), ("subscriber_index", []), ("event_id", []),
                             ("dispatch_sequence", True), ("finished_monotonic_ns", [])):
            with self.subTest(field=field):
                events, receipts, manifest = trace()
                receipts[0][field] = value
                self.assertEqual(audit_events(events, receipts, manifest)["status"], "fail")

    def test_phase_order_fails(self):
        events, receipts, manifest = trace()
        events[0]["payload"]["phase"], events[1]["payload"]["phase"] = "Attack-Agent", "Strategy-Agent"
        self.check_code(audit_events(events, receipts, manifest), "attack_before_strategy_phase")

    def test_run_identity_mismatch_fails(self):
        events, receipts, manifest = trace()
        events[0]["run_id"] = "other-run"
        self.check_code(audit_events(events, receipts, manifest), "event_run_mismatch")

    def test_finish_followed_by_cycle_event_fails(self):
        events, receipts, manifest = trace()
        events.append(copy.deepcopy(events[0]))
        self.check_code(audit_events(events, receipts, manifest), "cycle_event_after_run_finished")

    def test_parser_rejects_truncated_json_and_nonobjects(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.jsonl"
            for text in ('{"type":', "[]\n"):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_jsonl(path)

    def test_cli_hashes_sources_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events, receipts, manifest = trace()
            for name, rows in (("events", events), ("delivery_receipts", receipts)):
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = audit_run(root)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(result["sources"]), 3)
            output = root / "report.json"
            command = [sys.executable, "-B", "-X", "utf8",
                       str(Path(Experiment.__file__).with_name("audit_feedback_events.py")),
                       str(root), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", timeout=10)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", timeout=10)
            self.assertEqual(second.returncode, 1)
            self.assertEqual(output.read_bytes(), original)
            events.pop()  # Simulate an incomplete neutral run, without overwriting the report.
            (root / "events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
            incomplete = subprocess.run(command[:-2], capture_output=True, text=True,
                                        encoding="utf-8", timeout=10)
            self.assertEqual(incomplete.returncode, 2, incomplete.stderr)
            self.assertEqual(json.loads(incomplete.stdout)["status"], "incomplete")
            self.assertEqual(output.read_bytes(), original)

    def test_cli_roundtrips_unicode_source_paths(self):
        with tempfile.TemporaryDirectory(prefix="neutral-audit-\u00f6-\U0001f9ea-") as temporary:
            root = Path(temporary)
            events, receipts, manifest = trace(1)
            for name, rows in (("events", events), ("delivery_receipts", receipts)):
                (root / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            command = [sys.executable, "-B", "-X", "utf8",
                       str(Path(Experiment.__file__).with_name("audit_feedback_events.py")), str(root)]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding="utf-8", timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["sources"]["events"]["path"],
                             str((root / "events.jsonl").resolve()))


class DeliveryAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_neutral_roundtrip_passes_offline_audit(self):
        received = []

        class ReceiverA:
            def receive(self, message):
                received.append(("A", message.iteration))

        class ReceiverB:
            async def receive(self, message):
                await asyncio.sleep(0)
                received.append(("B", message.iteration))

        with tempfile.TemporaryDirectory() as temporary:
            store, bus = ArtifactStore(temporary, "neutral-roundtrip"), MessageBus()
            first, second = ReceiverA(), ReceiverB()
            bus.subscribe(MessageType.CRITIC_FEEDBACK, first.receive)
            bus.subscribe(MessageType.CRITIC_FEEDBACK, second.receive)
            bus.subscribe("*", store.event_handler)
            bus.set_delivery_observer(store.delivery_handler, {MessageType.CRITIC_FEEDBACK})
            store.write_manifest({"fixture": True, "model_calls": 0}, 0, 0)
            for iteration in (1, 2):
                for phase in ("Strategy-Agent", "Attack-Agent", "Critic-Agent"):
                    await bus.publish(Message(MessageType.PHASE_STARTED, "fixture", {"phase": phase}, iteration))
                await bus.publish(Message(MessageType.CRITIC_FEEDBACK, "fixture", {"document_id": "neutral"}, iteration))
                await bus.join()
                await bus.publish(Message(MessageType.Q_UPDATE, "fixture", {}, iteration))
                await bus.join()
            await bus.publish(Message(MessageType.RUN_FINISHED, "fixture", {"stop_reason": "fixture_complete"}, 2))
            await bus.close()
            store.close()
            result = audit_events(
                read_jsonl(store.run_dir / "events.jsonl"),
                read_jsonl(store.run_dir / "delivery_receipts.jsonl"),
                json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8")),
                (first.receive.__qualname__, second.receive.__qualname__, "ArtifactStore.event_handler"))
            self.assertEqual(result["status"], "pass", result)
            self.assertEqual(received, [("A", 1), ("B", 1), ("A", 2), ("B", 2)])

    async def test_sync_async_and_wildcard_order_is_unchanged(self):
        order, receipts = [], []
        bus = MessageBus()

        def first(message):
            order.append("sync")

        async def second(message):
            await asyncio.sleep(0)
            order.append("async")

        bus.subscribe(MessageType.CRITIC_FEEDBACK, first)
        bus.subscribe(MessageType.CRITIC_FEEDBACK, second)
        bus.subscribe("*", lambda message: order.append("wildcard"))
        bus.set_delivery_observer(receipts.append, {MessageType.CRITIC_FEEDBACK})
        message = Message(MessageType.CRITIC_FEEDBACK, "fixture", {"document_id": "文档"}, 1)
        await bus.publish(message)
        await bus.join()
        await bus.close()
        self.assertEqual(order, ["sync", "async", "wildcard"])
        self.assertEqual([item["subscriber_index"] for item in receipts], [0, 1, 2])
        self.assertTrue(all(item["status"] == "ok" for item in receipts))
        self.assertTrue(all(item["event_id"] == message.event_id for item in receipts))
        self.assertNotIn("document_id", json.dumps(receipts))
        self.assertEqual(bus.errors, [])

    async def test_repeated_message_is_not_suppressed(self):
        seen, receipts = [], []
        bus = MessageBus()
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append(message.event_id))
        bus.set_delivery_observer(receipts.append, {MessageType.CRITIC_FEEDBACK})
        message = Message(MessageType.CRITIC_FEEDBACK, "fixture", {})
        await bus.publish(message)
        await bus.publish(message)
        await bus.join()
        await bus.close()
        self.assertEqual(seen, [message.event_id, message.event_id])
        self.assertEqual([item["dispatch_sequence"] for item in receipts], [1, 2])

    async def test_handler_exception_does_not_skip_other_subscribers(self):
        seen, receipts = [], []
        bus = MessageBus()

        def failing(message):
            raise ValueError("fixture failure")

        bus.subscribe(MessageType.CRITIC_FEEDBACK, failing)
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append("received"))
        bus.set_delivery_observer(receipts.append, {MessageType.CRITIC_FEEDBACK})
        await bus.publish(Message(MessageType.CRITIC_FEEDBACK, "fixture", {}))
        await bus.close()
        self.assertEqual(seen, ["received"])
        self.assertEqual([item["status"] for item in receipts], ["error", "ok"])
        self.assertEqual(receipts[0]["error_type"], "ValueError")
        self.assertEqual(bus.errors, ["ValueError: fixture failure"])

    async def test_observer_failure_is_bounded_and_isolated(self):
        seen = []
        bus = MessageBus()

        def failing_observer(receipt):
            raise OSError("fixture failure")

        bus.set_delivery_observer(failing_observer, {MessageType.CRITIC_FEEDBACK})
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append(message))
        for _ in range(12):
            await bus.publish(Message(MessageType.CRITIC_FEEDBACK, "fixture", {}))
        await bus.close()
        self.assertEqual(len(seen), 12)
        self.assertEqual(bus.audit_error_count, 12)
        self.assertEqual(len(bus.audit_errors), 10)
        self.assertEqual(bus.errors, [])

    async def test_observer_cancellation_does_not_cancel_dispatch(self):
        bus, seen = MessageBus(), []

        def observer(receipt):
            raise asyncio.CancelledError()

        bus.set_delivery_observer(observer, {MessageType.CRITIC_FEEDBACK})
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append("first"))
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append("second"))
        await bus.publish(Message(MessageType.CRITIC_FEEDBACK, "fixture", {}))
        await bus.close()
        self.assertEqual(seen, ["first", "second"])
        self.assertEqual(bus.audit_error_count, 2)
        self.assertEqual(bus.errors, [])

    async def test_async_observer_is_rejected_without_changing_delivery(self):
        bus, seen = MessageBus(), []

        async def observer(receipt):
            seen.append("observer_should_not_run")

        bus.set_delivery_observer(observer, {MessageType.CRITIC_FEEDBACK})
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append("handler"))
        await bus.publish(Message(MessageType.CRITIC_FEEDBACK, "fixture", {}))
        await bus.close()
        self.assertEqual(seen, ["handler"])
        self.assertEqual(bus.audit_errors, ["TypeError"])

    async def test_cancelled_handler_still_propagates(self):
        bus, receipts, seen = MessageBus(), [], []

        async def cancelled(message):
            raise asyncio.CancelledError()

        bus.subscribe(MessageType.CRITIC_FEEDBACK, cancelled)
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: seen.append("unexpected"))
        bus.set_delivery_observer(receipts.append, {MessageType.CRITIC_FEEDBACK})
        await bus.publish(Message(MessageType.CRITIC_FEEDBACK, "fixture", {}))
        await bus.join()
        with self.assertRaises(asyncio.CancelledError):
            await bus._worker
        self.assertEqual(receipts[0]["status"], "interrupted")
        self.assertEqual(seen, [])

    async def test_only_selected_types_produce_receipts(self):
        bus, receipts = MessageBus(), []
        bus.subscribe("*", lambda message: None)
        bus.set_delivery_observer(receipts.append, {MessageType.CRITIC_FEEDBACK})
        await bus.publish(Message(MessageType.CHECKPOINT, "fixture", {}))
        await bus.close()
        self.assertEqual(receipts, [])

    async def test_metadata_is_snapshotted_before_callbacks(self):
        bus, receipts = MessageBus(), []

        def mutate(message):
            message.event_id, message.iteration, message.sender = "changed", 99, "changed"

        bus.subscribe(MessageType.CRITIC_FEEDBACK, mutate)
        bus.subscribe(MessageType.CRITIC_FEEDBACK, lambda message: None)
        bus.set_delivery_observer(receipts.append, {MessageType.CRITIC_FEEDBACK})
        message = Message(MessageType.CRITIC_FEEDBACK, "fixture", {}, 1)
        original_id = message.event_id
        await bus.publish(message)
        await bus.close()
        self.assertTrue(all(item["event_id"] == original_id and item["iteration"] == 1
                            and item["sender"] == "fixture" for item in receipts))

    async def test_artifact_metadata_and_receipt_persistence(self):
        with tempfile.TemporaryDirectory() as temporary:
            store, bus = ArtifactStore(temporary, "fixture"), MessageBus()
            bus.subscribe("*", store.event_handler)
            bus.set_delivery_observer(store.delivery_handler, {MessageType.CRITIC_FEEDBACK})
            store.write_manifest({"fixture": True}, 0, 0)
            message = Message(MessageType.CRITIC_FEEDBACK, "fixture", {"document_id": "文档"}, 1)
            await bus.publish(message)
            await bus.publish(Message(MessageType.CHECKPOINT, "fixture", {}, 1))
            await bus.close()
            store.close()
            events = read_jsonl(store.run_dir / "events.jsonl")
            receipts = read_jsonl(store.run_dir / "delivery_receipts.jsonl")
            manifest = json.loads((store.run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([item["sequence"] for item in events], [1, 2])
            self.assertEqual(events[0]["payload"]["document_id"], "文档")
            self.assertEqual(receipts[0]["subscriber"], "ArtifactStore.event_handler")
            self.assertEqual(receipts[0]["event_id"], message.event_id)
            self.assertEqual(receipts[0]["run_id"], manifest["run_id"])
            self.assertGreaterEqual(receipts[0]["duration_ms"], 0)
            self.assertNotIn("document_id", json.dumps(receipts))
            self.assertTrue(store._deliveries.closed)


class MessageIdentityTests(unittest.TestCase):
    def test_receipt_store_excludes_unexpected_payload_and_run_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary, "fixture")
            store.delivery_handler({"event_id": "fixture-event", "payload": {"private": "fixture"},
                                    "run_id": "wrong-run"})
            store.close()
            receipt = read_jsonl(store.run_dir / "delivery_receipts.jsonl")[0]
            self.assertEqual(receipt, {"event_id": "fixture-event", "run_id": store.run_id})

    def test_event_ids_are_unique_stable_and_positional_api_compatible(self):
        first = Message(MessageType.CHECKPOINT, "fixture", {}, 2, "fixture-time")
        second = Message(MessageType.CHECKPOINT, "fixture", {})
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertEqual(first.to_dict()["event_id"], first.to_dict()["event_id"])
        self.assertEqual(first.created_at, "fixture-time")
        self.assertEqual(first.iteration, 2)


if __name__ == "__main__":
    unittest.main()
