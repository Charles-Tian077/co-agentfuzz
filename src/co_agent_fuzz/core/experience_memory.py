from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _terms(text: str) -> Tuple[str, ...]:
    return tuple(sorted(set(_TOKEN_PATTERN.findall(text.casefold()))))


def _cosine(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / math.sqrt(len(left_set) * len(right_set))


@dataclass(frozen=True)
class ExperienceRecord:
    iteration: int
    context_key: str
    context_terms: Tuple[str, ...]
    seed_id: str
    action: str
    tier: str
    outcome: str
    score: float
    reward: float


@dataclass(frozen=True)
class RetrievedExperience:
    action: str
    tier: str
    outcome: str
    similarity: float
    reward: float


class StrategyMemory:
    """Compact ASTRA-style three-tier memory with zero-model-cost retrieval."""

    def __init__(self, enabled: bool = True, max_size: int = 2000,
                 top_k: int = 5, min_similarity: float = 0.05,
                 tier_weights: Dict[str, float] | None = None):
        self.enabled = enabled
        self.max_size = max(1, max_size)
        self.top_k = max(1, top_k)
        self.min_similarity = min_similarity
        self.tier_weights = tier_weights or {
            "effective": 1.0,
            "promising": 0.35,
            "ineffective": -0.40,
        }
        self.records: List[ExperienceRecord] = []

    def add(self, *, iteration: int, context_key: str, context_text: str,
            seed_id: str, action: str, tier: str, outcome: str,
            score: float, reward: float) -> None:
        if not self.enabled:
            return
        self.records.append(ExperienceRecord(
            iteration=iteration,
            context_key=context_key,
            context_terms=_terms(context_text),
            seed_id=seed_id,
            action=action,
            tier=tier,
            outcome=outcome,
            score=float(score),
            reward=float(reward),
        ))
        overflow = len(self.records) - self.max_size
        if overflow > 0:
            del self.records[:overflow]

    def retrieve(self, context_text: str, active_actions: Iterable[str]) -> tuple[Dict[str, float], List[RetrievedExperience]]:
        if not self.enabled or not self.records:
            return {}, []
        active = set(active_actions)
        query_terms = _terms(context_text)
        ranked = []
        for record in self.records:
            if record.action not in active:
                continue
            similarity = _cosine(query_terms, record.context_terms)
            if similarity >= self.min_similarity:
                ranked.append((similarity, record.iteration, record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = ranked[:self.top_k]

        evidence: List[RetrievedExperience] = []
        totals: Dict[str, float] = {}
        weights: Dict[str, float] = {}
        for similarity, _, record in selected:
            tier_weight = float(self.tier_weights.get(record.tier, 0.0))
            strength = similarity * (0.5 + 0.5 * min(1.0, abs(record.reward)))
            totals[record.action] = totals.get(record.action, 0.0) + tier_weight * strength
            weights[record.action] = weights.get(record.action, 0.0) + similarity
            evidence.append(RetrievedExperience(
                action=record.action,
                tier=record.tier,
                outcome=record.outcome,
                similarity=similarity,
                reward=record.reward,
            ))
        biases = {
            action: totals[action] / max(weights[action], 1e-12)
            for action in totals
        }
        return biases, evidence

    @property
    def tier_counts(self) -> Dict[str, int]:
        counts = {"effective": 0, "promising": 0, "ineffective": 0}
        for record in self.records:
            counts[record.tier] = counts.get(record.tier, 0) + 1
        return counts

    def load(self, path: str | Path) -> int:
        if not self.enabled:
            return 0
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded = 0
        for item in payload.get("records", []):
            try:
                self.records.append(ExperienceRecord(
                    iteration=int(item["iteration"]),
                    context_key=str(item["context_key"]),
                    context_terms=tuple(str(term) for term in item["context_terms"]),
                    seed_id=str(item["seed_id"]),
                    action=str(item["action"]),
                    tier=str(item["tier"]),
                    outcome=str(item["outcome"]),
                    score=float(item["score"]),
                    reward=float(item["reward"]),
                ))
                loaded += 1
            except (KeyError, TypeError, ValueError):
                continue
        self.records = self.records[-self.max_size:]
        return loaded

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "max_size": self.max_size,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "tier_weights": self.tier_weights,
            "tier_counts": self.tier_counts,
            "records": [asdict(record) for record in self.records],
        }
