from __future__ import annotations

from typing import Dict, Iterable, List

from .adversarial import Base64WrapperMutator, GeneticCrossoverMutator
from .base import PromptMutator
from .char_level import TokenSwapMutator, UnicodeConfusableMutator, ZeroWidthInsertionMutator
from .semantic import ContextualizationMutator, MultilingualWrapperMutator, RolePlayMutator
from .word_level import (
    DefinitionReplacementMutator, DependencyExpansionMutator,
    InstructionInjectionMutator, SentenceReorderMutator, SynonymReplacementMutator,
)


_ACTION_TAGS = {
    "unicode_confusable": {"character", "obfuscation"},
    "zero_width_insertion": {"character", "obfuscation"},
    "token_swap": {"character", "order"},
    "synonym_replacement": {"lexical", "meaning"},
    "definition_replacement": {"lexical", "meaning"},
    "instruction_injection": {"lexical", "priority"},
    "sentence_reorder": {"syntactic", "order"},
    "dependency_expansion": {"syntactic", "meaning"},
    "role_play": {"semantic", "context"},
    "contextualization": {"semantic", "context"},
    "multilingual_wrapper": {"semantic", "multilingual"},
    "base64_wrapper": {"adversarial", "encoding"},
    "genetic_crossover": {"adversarial", "composition"},
}


class MutatorRegistry:
    def __init__(self):
        self._mutators: Dict[str, PromptMutator] = {}

    def register(self, mutator: PromptMutator) -> None:
        if mutator.name in self._mutators:
            raise ValueError(f"Mutator {mutator.name!r} is already registered")
        self._mutators[mutator.name] = mutator

    def get(self, name: str) -> PromptMutator:
        return self._mutators[name]

    @property
    def names(self) -> List[str]:
        return list(self._mutators)

    def select(self, names: Iterable[str]) -> "MutatorRegistry":
        selected = MutatorRegistry()
        for name in names:
            selected.register(self.get(name))
        return selected

    @property
    def action_tags(self) -> Dict[str, set[str]]:
        return {
            name: set(_ACTION_TAGS.get(name, ())) | {mutator.family}
            for name, mutator in self._mutators.items()
        }


def build_default_registry() -> MutatorRegistry:
    registry = MutatorRegistry()
    for mutator in (
        UnicodeConfusableMutator(), ZeroWidthInsertionMutator(), TokenSwapMutator(),
        SynonymReplacementMutator(), DefinitionReplacementMutator(),
        InstructionInjectionMutator(), SentenceReorderMutator(), DependencyExpansionMutator(),
        RolePlayMutator(), ContextualizationMutator(), MultilingualWrapperMutator(),
        Base64WrapperMutator(), GeneticCrossoverMutator(),
    ):
        registry.register(mutator)
    return registry
