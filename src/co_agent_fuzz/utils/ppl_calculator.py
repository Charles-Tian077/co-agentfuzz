from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class UnigramPerplexity:
    """Cheap deterministic seed-corpus PPL proxy used before optional neural filters."""

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.counts: Counter[str] = Counter()
        self.total = 0
        self.vocab_size = 1

    def fit(self, texts: Iterable[str]) -> "UnigramPerplexity":
        for text in texts:
            self.counts.update(_tokens(text))
        self.total = sum(self.counts.values())
        self.vocab_size = max(1, len(self.counts) + 1)
        return self

    def perplexity(self, text: str) -> float:
        tokens = _tokens(text)
        if not tokens or self.total == 0:
            return float("inf")
        denominator = self.total + self.smoothing * self.vocab_size
        negative_log_likelihood = 0.0
        for token in tokens:
            probability = (self.counts.get(token, 0) + self.smoothing) / denominator
            negative_log_likelihood -= math.log(probability)
        return math.exp(negative_log_likelihood / len(tokens))
