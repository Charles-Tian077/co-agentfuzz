from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass
class MutationContext:
    mode: str
    peer_seeds: List[str] = field(default_factory=list)
    access_goal: str = "Access granted"


class PromptMutator(Protocol):
    name: str
    family: str

    def mutate(self, seed: str, context: MutationContext,
               rng: random.Random) -> str:
        ...
