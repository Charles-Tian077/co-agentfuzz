from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

from co_agent_fuzz.core.experience_memory import StrategyMemory
from co_agent_fuzz.models import CandidateRecord, SeedRecord


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def semantic_coverage(source: str, candidate: str) -> float:
    """Recall of source concepts; wrappers are not penalised for adding context."""
    source_terms = set(_TOKEN_PATTERN.findall(source.casefold()))
    candidate_terms = set(_TOKEN_PATTERN.findall(candidate.casefold()))
    if not source_terms:
        return 1.0
    return len(source_terms & candidate_terms) / len(source_terms)


@dataclass(frozen=True)
class CriticFeedback:
    mean_reward: float
    dominant_outcome: str
    outcome_counts: Dict[str, int]
    rollbacks: int
    tiers: Dict[str, int]


class CriticAgent:
    """Stage 3: classify feedback, gate drifted seeds, and update experience."""

    _priority = {
        "success": 5,
        "near_miss": 4,
        "rejection": 3,
        "semantic_deviation": 2,
        "malformed": 1,
        "target_error": 0,
    }

    def __init__(self, memory: StrategyMemory, config: Dict):
        self.memory = memory
        self.config = config

    def review(self, *, iteration: int, context_key: str, context_text: str,
               seed: SeedRecord, action: str,
               candidates: Iterable[CandidateRecord]) -> CriticFeedback:
        rewards = self.config.get("rewards", {})
        promising = float(self.config.get("promising_threshold", 0.35))
        semantic_threshold = float(self.config.get("semantic_similarity_threshold", 0.35))
        outcomes: Counter[str] = Counter()
        tiers: Counter[str] = Counter()
        dense_rewards: List[float] = []
        rollbacks = 0

        for candidate in candidates:
            similarity = semantic_coverage(seed.text, candidate.variant.text)
            if not candidate.filter.accepted:
                outcome = "malformed"
            elif candidate.evaluations and all(item.error for item in candidate.evaluations):
                outcome = "target_error"
            elif similarity < semantic_threshold:
                outcome = "semantic_deviation"
                rollbacks += 1
            elif candidate.hard_successes:
                outcome = "success"
            elif candidate.oracle_successes or candidate.mean_score >= promising:
                outcome = "near_miss"
            else:
                outcome = "rejection"

            if outcome == "success":
                tier = "effective"
            elif outcome == "near_miss":
                tier = "promising"
            else:
                tier = "ineffective"
            reward = float(rewards.get(outcome, 0.0))
            critic_score = 1.0 if outcome == "success" else 0.5 if outcome == "near_miss" else 0.0
            candidate.variant.metadata.update({
                "critic_outcome": outcome,
                "memory_tier": tier,
                "semantic_coverage": similarity,
                "rollback": outcome == "semantic_deviation",
                "dense_reward": reward,
                "critic_score": critic_score,
            })
            outcomes[outcome] += 1
            tiers[tier] += 1
            dense_rewards.append(reward)
            self.memory.add(
                iteration=iteration,
                context_key=context_key,
                context_text=context_text,
                seed_id=seed.seed_id,
                action=action,
                tier=tier,
                outcome=outcome,
                score=candidate.mean_score,
                reward=reward,
            )

        dominant = max(outcomes, key=lambda item: (self._priority.get(item, -1), outcomes[item])) if outcomes else "empty"
        return CriticFeedback(
            mean_reward=sum(dense_rewards) / len(dense_rewards) if dense_rewards else 0.0,
            dominant_outcome=dominant,
            outcome_counts=dict(outcomes),
            rollbacks=rollbacks,
            tiers=dict(tiers),
        )
