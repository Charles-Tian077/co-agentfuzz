from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from co_agent_fuzz.models import SeedRecord


class OnlineRandomForestSurrogate:
    """Online-refitted success surrogate for seed/mutator prioritisation."""

    def __init__(self, actions: Iterable[str], enabled: bool = True,
                 min_samples: int = 30, retrain_interval: int = 10,
                 random_seed: int = 0):
        self.actions = list(actions)
        self.enabled = enabled
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval
        self._action_index = {name: index for index, name in enumerate(self.actions)}
        self._x: List[List[float]] = []
        self._y: List[int] = []
        self._model = RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced",
            random_state=random_seed, n_jobs=1)
        self.ready = False

    def _features(self, text: str, seed: SeedRecord, action: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        characters = max(1, len(text))
        keyword_count = sum(text.lower().count(word) for word in (
            "ignore", "instruction", "system", "access", "decode", "role", "priority"))
        features = [
            min(len(text), 4096) / 4096.0,
            min(len(tokens), 512) / 512.0,
            len(set(tokens)) / max(1, len(tokens)),
            sum(ord(char) > 127 for char in text) / characters,
            sum(not char.isprintable() for char in text) / characters,
            min(keyword_count, 20) / 20.0,
            seed.asr,
            seed.mean_score,
            min(seed.generation, 20) / 20.0,
        ]
        one_hot = [0.0] * len(self.actions)
        if action in self._action_index:
            one_hot[self._action_index[action]] = 1.0
        return features + one_hot

    def observe(self, text: str, seed: SeedRecord, action: str, success: bool) -> None:
        if not self.enabled:
            return
        self._x.append(self._features(text, seed, action))
        self._y.append(int(success))
        if (len(self._x) >= self.min_samples and len(set(self._y)) > 1
                and (len(self._x) - self.min_samples) % self.retrain_interval == 0):
            self._model.fit(np.asarray(self._x), np.asarray(self._y))
            self.ready = True

    def predict(self, text: str, seed: SeedRecord, action: str) -> Optional[float]:
        if not self.enabled or not self.ready:
            return None
        probabilities = self._model.predict_proba(
            np.asarray([self._features(text, seed, action)]))[0]
        classes = list(self._model.classes_)
        return float(probabilities[classes.index(1)]) if 1 in classes else 0.0

    def action_bias(self, seed: SeedRecord, weight: float = 0.2) -> Dict[str, float]:
        if not self.ready:
            return {}
        return {
            action: weight * (self.predict(seed.text, seed, action) or 0.0)
            for action in self.actions
        }
