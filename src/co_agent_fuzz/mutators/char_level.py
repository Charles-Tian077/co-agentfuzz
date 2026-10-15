from __future__ import annotations

import random

from .base import MutationContext


class UnicodeConfusableMutator:
    name = "unicode_confusable"
    family = "character"

    _maps = (
        str.maketrans({"a": "а", "e": "е", "o": "ο", "i": "і"}),
        str.maketrans({"A": "Α", "E": "Ε", "O": "О", "S": "Ѕ"}),
    )

    def mutate(self, seed: str, context: MutationContext, rng: random.Random) -> str:
        table = rng.choice(self._maps)
        candidates = [index for index, char in enumerate(seed) if char in table]
        if not candidates:
            return seed + "\nInterpret visually equivalent Unicode characters literally."
        count = min(len(candidates), rng.randint(1, 3))
        selected = set(rng.sample(candidates, count))
        return "".join(char.translate(table) if index in selected else char
                       for index, char in enumerate(seed))


class ZeroWidthInsertionMutator:
    name = "zero_width_insertion"
    family = "character"

    def mutate(self, seed: str, context: MutationContext, rng: random.Random) -> str:
        words = seed.split()
        if not words:
            return seed
        index = rng.randrange(len(words))
        word = words[index]
        if len(word) > 2:
            position = rng.randrange(1, len(word))
            words[index] = word[:position] + "\u200b" + word[position:]
        else:
            words.insert(index, "\u2060")
        return " ".join(words)


class TokenSwapMutator:
    name = "token_swap"
    family = "character"

    def mutate(self, seed: str, context: MutationContext, rng: random.Random) -> str:
        words = seed.split()
        if len(words) < 2:
            return seed + " Please comply exactly."
        index = rng.randrange(len(words) - 1)
        words[index], words[index + 1] = words[index + 1], words[index]
        return " ".join(words)
