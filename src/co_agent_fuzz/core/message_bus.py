from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from uuid import uuid4


class MessageType(str, Enum):
    PHASE_STARTED = "PHASE_STARTED"
    SEED_SUBMIT = "SEED_SUBMIT"
    STRATEGY_RETRIEVED = "STRATEGY_RETRIEVED"
    VARIANT_GENERATED = "VARIANT_GENERATED"
    HELPER_REQUEST = "HELPER_REQUEST"
    HELPER_RESPONSE = "HELPER_RESPONSE"
    FILTER_RESULT = "FILTER_RESULT"
    TARGET_RESPONSE = "TARGET_RESPONSE"
    ORACLE_RESULT = "ORACLE_RESULT"
    CRITIC_FEEDBACK = "CRITIC_FEEDBACK"
    ROLLBACK = "ROLLBACK"
    Q_UPDATE = "Q_UPDATE"
    SEED_EXPANDED = "SEED_EXPANDED"
    CHECKPOINT = "CHECKPOINT"
    RUN_FINISHED = "RUN_FINISHED"


@dataclass
class Message:
    type: MessageType
    sender: str
    payload: Dict[str, Any]
    iteration: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


MessageHandler = Callable[[Message], Optional[Awaitable[None]]]
DeliveryObserver = Callable[[Dict[str, Any]], None]


class MessageBus:
    """Small asynchronous in-memory bus used for decoupling and audit events."""

    def __init__(self, max_queue_size: int = 1000):
        self._queue: asyncio.Queue[Optional[Message]] = asyncio.Queue(max_queue_size)
        self._subscribers: Dict[str, List[MessageHandler]] = {}
        self._worker: Optional[asyncio.Task[None]] = None
        self.errors: List[str] = []
        self.audit_errors: List[str] = []
        self.audit_error_count = 0
        self._delivery_observer: Optional[DeliveryObserver] = None
        self._observed_types: set[str] = set()
        self._dispatch_sequence = 0

    def set_delivery_observer(self, observer: DeliveryObserver,
                              message_types: Iterable[MessageType | str]) -> None:
        """Add metadata-only receipts; never retry, deduplicate or alter delivery."""
        self._delivery_observer = observer
        self._observed_types = {
            item.value if isinstance(item, MessageType) else item
            for item in message_types
        }

    def _record_delivery(self, receipt: Dict[str, Any]) -> None:
        observer = self._delivery_observer
        if observer is None:
            return
        try:
            result = observer(receipt)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Delivery observers must be synchronous")
        except (Exception, asyncio.CancelledError) as exc:
            # Audit failures remain separate from the existing run-error policy.
            self.audit_error_count += 1
            if len(self.audit_errors) < 10:
                self.audit_errors.append(type(exc).__name__)

    def subscribe(self, message_type: MessageType | str, handler: MessageHandler) -> None:
        key = message_type.value if isinstance(message_type, MessageType) else message_type
        self._subscribers.setdefault(key, []).append(handler)

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="co_agent_fuzz-message-bus")

    async def publish(self, message: Message) -> None:
        if self._worker is None:
            await self.start()
        await self._queue.put(message)

    async def join(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        if self._worker is None:
            return
        await self._queue.join()
        await self._queue.put(None)
        await self._worker
        self._worker = None

    async def _run(self) -> None:
        while True:
            message = await self._queue.get()
            try:
                if message is None:
                    return
                self._dispatch_sequence += 1
                observed = (self._delivery_observer is not None
                            and message.type.value in self._observed_types)
                metadata = {
                    "event_id": message.event_id, "type": message.type.value,
                    "sender": message.sender, "iteration": message.iteration,
                    "dispatch_sequence": self._dispatch_sequence,
                } if observed else None
                handlers = self._subscribers.get(message.type.value, []) + self._subscribers.get("*", [])
                for index, handler in enumerate(handlers):
                    receipt = None
                    if observed:
                        receipt = {
                            **metadata, "subscriber_index": index,
                            "subscriber": getattr(handler, "__qualname__", type(handler).__qualname__),
                            "started_at": datetime.now(timezone.utc).isoformat(),
                            "started_monotonic_ns": time.monotonic_ns(),
                            "status": "interrupted", "error_type": None,
                        }
                    try:
                        result = handler(message)
                        if inspect.isawaitable(result):
                            await result
                        if receipt is not None:
                            receipt["status"] = "ok"
                    except Exception as exc:  # Event observers must not kill a fuzz run.
                        self.errors.append(f"{type(exc).__name__}: {exc}")
                        if receipt is not None:
                            receipt.update(status="error", error_type=type(exc).__name__)
                    finally:
                        if receipt is not None:
                            receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
                            receipt["finished_monotonic_ns"] = time.monotonic_ns()
                            receipt["duration_ms"] = (
                                receipt["finished_monotonic_ns"]
                                - receipt["started_monotonic_ns"]) / 1_000_000
                            self._record_delivery(receipt)
            finally:
                self._queue.task_done()
