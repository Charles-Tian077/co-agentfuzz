from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from co_agent_fuzz.models import SeedRecord, stable_id


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class SeedPool:
    def __init__(self, max_size: int = 10000, ucb_exploration: float = 1.4,
                 depth_penalty: float = 0.01,
                 feedback_exploitation_rate: float = 0.40,
                 feedback_min_trials: int = 1,
                 random_seed: int = 0):
        self.max_size = max_size
        self.ucb_exploration = ucb_exploration
        self.depth_penalty = depth_penalty
        self.feedback_exploitation_rate = feedback_exploitation_rate
        self.feedback_min_trials = max(1, feedback_min_trials)
        self._rng = random.Random(random_seed)
        self._records: Dict[str, SeedRecord] = {}
        self._text_index: Dict[str, str] = {}
        self.total_selections = 0
        self.last_selection_reason = "initial"

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[SeedRecord]:
        return list(self._records.values())

    def get(self, seed_id: str) -> SeedRecord:
        return self._records[seed_id]

    def add(self, text: str, parent_id: Optional[str] = None, source: str = "initial",
            generation: int = 0, iteration: int = 0) -> Tuple[SeedRecord, bool]:
        cleaned = _normalise(text)
        if not cleaned:
            raise ValueError("Seed text cannot be empty")
        if cleaned in self._text_index:
            return self._records[self._text_index[cleaned]], False

        seed_id = stable_id("seed", cleaned)
        record = SeedRecord(
            seed_id=seed_id,
            text=cleaned,
            parent_id=parent_id,
            source=source,
            generation=generation,
            created_iteration=iteration,
        )
        self._records[seed_id] = record
        self._text_index[cleaned] = seed_id
        return record, True

    def add_many(self, texts: Iterable[str], source: str = "initial") -> int:
        added = 0
        for text in texts:
            _, is_new = self.add(text, source=source)
            added += int(is_new)
        return added

    def select_ucb(self, iteration: int) -> SeedRecord:
        if not self._records:
            raise RuntimeError("Seed pool is empty")

        records = list(self._records.values())
        evaluated = [item for item in records if item.trials >= self.feedback_min_trials]
        unseen = [item for item in records if item.selections == 0 and item.trials == 0]
        use_feedback = bool(evaluated) and (
            not unseen or self._rng.random() < self.feedback_exploitation_rate)

        if use_feedback:
            log_total = math.log(max(2, self.total_selections + 1))

            def feedback_score(item: SeedRecord) -> float:
                quality = 0.65 * item.asr + 0.35 * item.mean_score
                uncertainty = self.ucb_exploration * math.sqrt(
                    log_total / max(1, item.trials))
                return quality + uncertainty - self.depth_penalty * item.generation

            best = max(feedback_score(item) for item in evaluated)
            candidates = [
                item for item in evaluated
                if abs(feedback_score(item) - best) < 1e-12
            ]
            chosen = self._rng.choice(candidates)
            self.last_selection_reason = "feedback_exploit"
        elif unseen:
            chosen = self._rng.choice(unseen)
            self.last_selection_reason = "unseen_explore"
        else:
            log_total = math.log(max(2, self.total_selections))

            def score(item: SeedRecord) -> float:
                exploit = 0.5 * item.asr + 0.5 * item.mean_score
                explore = self.ucb_exploration * math.sqrt(log_total / item.selections)
                return exploit + explore - self.depth_penalty * item.generation

            best = max(score(item) for item in self._records.values())
            candidates = [item for item in self._records.values() if abs(score(item) - best) < 1e-12]
            chosen = self._rng.choice(candidates)
            self.last_selection_reason = "ucb"

        chosen.selections += 1
        chosen.last_selected_iteration = iteration
        self.total_selections += 1
        return chosen

    def update(self, seed_id: str, trials: int, hard_successes: int,
               soft_score_sum: float) -> None:
        record = self._records[seed_id]
        record.trials += trials
        record.hard_successes += hard_successes
        record.soft_score_sum += soft_score_sum

    def prune(self, min_asr: float = 0.0) -> List[str]:
        if len(self._records) <= self.max_size:
            return []
        protected = {item.seed_id for item in self._records.values() if item.source == "initial"}
        removable = [item for item in self._records.values() if item.seed_id not in protected]
        removable.sort(
            key=lambda item: (item.asr >= min_asr, item.mean_score, item.asr,
                              item.last_selected_iteration),
            reverse=True,
        )
        keep_budget = max(0, self.max_size - len(protected))
        keep = protected | {item.seed_id for item in removable[:keep_budget]}
        removed = [seed_id for seed_id in self._records if seed_id not in keep]
        for seed_id in removed:
            text = self._records[seed_id].text
            del self._records[seed_id]
            self._text_index.pop(text, None)
        return removed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "total_selections": self.total_selections,
            "feedback_exploitation_rate": self.feedback_exploitation_rate,
            "feedback_min_trials": self.feedback_min_trials,
            "last_selection_reason": self.last_selection_reason,
            "seeds": [item.to_dict() for item in self._records.values()],
        }
