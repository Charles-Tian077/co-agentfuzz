from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from co_agent_fuzz.core.message_bus import Message
from co_agent_fuzz.models import CandidateRecord, RunStats


DELIVERY_FIELDS = (
    "event_id", "type", "sender", "iteration", "dispatch_sequence",
    "subscriber_index", "subscriber", "started_at", "started_monotonic_ns",
    "finished_at", "finished_monotonic_ns", "duration_ms", "status", "error_type",
)


class ArtifactStore:
    def __init__(self, output_root: str, run_name: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = run_name or f"run-{timestamp}"
        self.run_dir = Path(output_root).resolve() / safe_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.run_id = uuid4().hex
        self._event_sequence = 0
        self._events = (self.run_dir / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self._deliveries = (self.run_dir / "delivery_receipts.jsonl").open(
            "w", encoding="utf-8", buffering=1)
        self._candidates = (self.run_dir / "candidates.jsonl").open("w", encoding="utf-8", buffering=1)
        self._metrics = (self.run_dir / "metrics.jsonl").open("w", encoding="utf-8", buffering=1)
        self._success_file = (self.run_dir / "successes.csv").open(
            "w", encoding="utf-8-sig", newline="", buffering=1)
        self._success_writer = csv.writer(self._success_file)
        self._success_writer.writerow([
            "iteration", "variant_id", "seed_id", "mutator", "defense_id",
            "benchmark_success", "oracle_success", "score", "oracle_layer", "prompt", "response",
        ])

    @staticmethod
    def _dump_line(handle, payload: Dict[str, Any]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def event_handler(self, message: Message) -> None:
        self._event_sequence += 1
        self._dump_line(self._events, {
            **message.to_dict(), "run_id": self.run_id,
            "sequence": self._event_sequence,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "recorded_monotonic_ns": time.monotonic_ns(),
        })

    def delivery_handler(self, receipt: Dict[str, Any]) -> None:
        metadata = {key: receipt[key] for key in DELIVERY_FIELDS if key in receipt}
        self._dump_line(self._deliveries, {**metadata, "run_id": self.run_id})

    def write_candidate(self, candidate: CandidateRecord) -> None:
        self._dump_line(self._candidates, candidate.to_dict())
        for evaluation in candidate.evaluations:
            if evaluation.oracle.benchmark_success or evaluation.oracle.hard_success:
                self._success_writer.writerow([
                    candidate.variant.iteration,
                    candidate.variant.variant_id,
                    candidate.variant.seed_id,
                    candidate.variant.mutator,
                    evaluation.defense_id,
                    evaluation.oracle.benchmark_success,
                    evaluation.oracle.hard_success,
                    evaluation.oracle.score,
                    evaluation.oracle.layer,
                    candidate.variant.text,
                    evaluation.response,
                ])

    def write_metric(self, payload: Dict[str, Any]) -> None:
        self._dump_line(self._metrics, payload)

    def write_manifest(self, config: Dict[str, Any], seed_count: int,
                       defense_count: int) -> None:
        packages = {}
        for package in ("openai", "httpx", "numpy", "pandas", "scikit-learn", "PyYAML"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        manifest = {
            "run_id": self.run_id,
            "audit_schema_version": 1,
            "delivery_trace": "delivery_receipts.jsonl",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "seed_count": seed_count,
            "defense_count": defense_count,
            "config": config,
        }
        self.write_json("manifest.json", manifest)
        with (self.run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    def checkpoint(self, stats: RunStats, q_table: Dict[str, Any],
                   seed_pool: Dict[str, Any], experience_memory: Optional[Dict[str, Any]] = None) -> None:
        self.write_json("summary.json", stats.to_dict())
        self.write_json("q_table.json", q_table)
        self.write_json("seed_pool.json", seed_pool)
        if experience_memory is not None:
            self.write_json("experience_memory.json", experience_memory)

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        destination = self.run_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.replace(destination)

    def close(self) -> None:
        for handle in (self._events, self._deliveries, self._candidates,
                       self._metrics, self._success_file):
            handle.close()
