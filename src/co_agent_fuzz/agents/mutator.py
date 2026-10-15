from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Optional

from co_agent_fuzz.core.message_bus import Message
from co_agent_fuzz.core.surrogate import OnlineRandomForestSurrogate
from co_agent_fuzz.models import SeedRecord, VariantRecord, stable_id
from co_agent_fuzz.mutators import MutationContext, MutatorRegistry


class MutatorAgent:
    def __init__(self, registry: MutatorRegistry, random_seed: int = 0,
                 surrogate: Optional[OnlineRandomForestSurrogate] = None,
                 oversample_factor: int = 1, feedback_ema_alpha: float = 0.30):
        self.registry = registry
        self.surrogate = surrogate
        self.oversample_factor = max(1, int(oversample_factor))
        self.feedback_ema_alpha = float(feedback_ema_alpha)
        self._rng = random.Random(random_seed)
        self.feedback_scores: Dict[str, float] = {
            name: 0.0 for name in registry.names
        }
        self.feedback_counts: Dict[str, int] = {
            name: 0 for name in registry.names
        }
        self.feedback_updates = 0
        self.last_proposal_count = 0

    def on_critic_feedback(self, message: Message) -> None:
        """Adapt proposal ranking from the Critic's latest reward."""
        action = str(message.payload.get("action", ""))
        if action not in self.feedback_scores:
            return
        reward = float(message.payload.get("mean_reward", 0.0))
        old = self.feedback_scores[action]
        alpha = self.feedback_ema_alpha
        self.feedback_scores[action] = old + alpha * (reward - old)
        self.feedback_counts[action] += 1
        self.feedback_updates += 1

    def generate(self, seed: SeedRecord, mutator_name: str, count: int,
                 iteration: int, peer_seeds: Iterable[str], mode: str) -> List[VariantRecord]:
        mutator = self.registry.get(mutator_name)
        context = MutationContext(mode=mode, peer_seeds=list(peer_seeds))
        variants: List[VariantRecord] = []
        seen = set()
        proposal_target = max(count, count * self.oversample_factor)
        attempts = max(proposal_target * 3, proposal_target)
        for attempt in range(attempts):
            text = mutator.mutate(seed.text, context, self._rng).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            variants.append(VariantRecord(
                variant_id=stable_id("variant", seed.seed_id, mutator_name, iteration, attempt, text),
                seed_id=seed.seed_id,
                text=text,
                mutator=mutator_name,
                generation=seed.generation + 1,
                iteration=iteration,
                metadata={"family": mutator.family},
            ))
            if len(variants) >= proposal_target:
                break
        self.last_proposal_count = len(variants)
        if len(variants) <= count:
            return variants
        return self._rank(variants, seed, mutator_name)[:count]

    def _rank(self, variants: List[VariantRecord], seed: SeedRecord,
              action: str) -> List[VariantRecord]:
        if not self.feedback_counts[action] and not (
                self.surrogate is not None and self.surrogate.ready):
            return variants

        seed_terms = set(re.findall(r"\w+", seed.text.casefold(), flags=re.UNICODE))
        feedback = self.feedback_scores[action]
        ranked = []
        for order, variant in enumerate(variants):
            terms = set(re.findall(r"\w+", variant.text.casefold(), flags=re.UNICODE))
            coverage = len(seed_terms & terms) / max(1, len(seed_terms))
            novelty = len(terms - seed_terms) / max(1, len(terms | seed_terms))
            if feedback >= 0:
                adaptive_score = coverage
            else:
                adaptive_score = coverage * (0.5 + 0.5 * novelty)
            predicted = (
                self.surrogate.predict(variant.text, seed, action)
                if self.surrogate is not None else None
            )
            score = adaptive_score if predicted is None else 0.75 * predicted + 0.25 * adaptive_score
            variant.metadata.update({
                "proposal_score": score,
                "surrogate_generation_score": predicted,
                "operator_feedback": feedback,
            })
            ranked.append((score, -order, variant))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked]
