from __future__ import annotations

import random
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


@dataclass
class ActionStats:
    trials: int = 0
    hard_successes: int = 0
    soft_score_sum: float = 0.0

    @property
    def asr(self) -> float:
        return self.hard_successes / self.trials if self.trials else 0.0


class QLearningEngine:
    def __init__(self, actions: Iterable[str], learning_rate: float = 0.1,
                 discount_factor: float = 0.9, epsilon_start: float = 0.5,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 bins: Tuple[float, float] = (0.1, 0.4),
                 reward_weights: Tuple[float, float, float] = (1.0, 0.5, 0.1),
                 dense_reward_weight: float = 0.35,
                 similarity_credit: float = 0.20,
                 global_credit_weight: float = 0.35,
                 action_tags: Mapping[str, Iterable[str]] | None = None,
                 random_seed: int = 0):
        self.actions = list(actions)
        if not self.actions:
            raise ValueError("At least one mutator action is required")
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.bins = bins
        self.reward_weights = reward_weights
        self.dense_reward_weight = dense_reward_weight
        self.similarity_credit = similarity_credit
        self.global_credit_weight = global_credit_weight
        self.action_tags = {
            action: set((action_tags or {}).get(action, ())) for action in self.actions
        }
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.global_q: Dict[str, float] = {action: 0.0 for action in self.actions}
        self.stats: Dict[str, ActionStats] = {action: ActionStats() for action in self.actions}
        self._rng = random.Random(random_seed)

    def _bin(self, value: float) -> int:
        if value < self.bins[0]:
            return 0
        if value < self.bins[1]:
            return 1
        return 2

    def state(self, seed_asr: float, action: str, context: str = "global") -> str:
        return f"{context}|{self._bin(seed_asr)}|{self._bin(self.stats[action].asr)}"

    def q_value(self, state: str, action: str) -> float:
        return self.q_table[state].get(action, 0.0)

    def select(self, seed_asr: float, active_actions: Iterable[str],
               action_bias: Mapping[str, float] | None = None, *,
               context: str = "global", selection_policy: str = "epsilon_greedy",
               temperature: float = 0.15, q_smoothing: float = 0.01) -> Tuple[str, str, bool]:
        active = [action for action in active_actions if action in self.stats]
        if not active:
            active = list(self.actions)
        explored = self._rng.random() < self.epsilon
        if explored:
            action = self._rng.choice(active)
        else:
            bias = action_bias or {}
            values = {
                candidate: (
                    self.q_value(self.state(seed_asr, candidate, context), candidate)
                    if candidate in self.q_table[self.state(seed_asr, candidate, context)]
                    else q_smoothing
                )
                + self.global_credit_weight * self.global_q[candidate]
                + float(bias.get(candidate, 0.0))
                for candidate in active
            }
            if selection_policy == "softmax" and temperature > 0:
                largest = max(values.values())
                weights = {
                    candidate: math.exp((value - largest) / temperature)
                    for candidate, value in values.items()
                }
                action = self._rng.choices(
                    list(weights), weights=list(weights.values()), k=1)[0]
            else:
                best = max(values.values())
                action = self._rng.choice(
                    [candidate for candidate, value in values.items() if abs(value - best) < 1e-12])
        state = self.state(seed_asr, action, context)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return action, state, explored

    def update(self, state: str, action: str, next_seed_asr: float,
               seed_delta: float, mutator_delta: float, normalised_cost: float,
               trials: int, hard_successes: int, soft_score_sum: float,
               dense_reward: float = 0.0, context: str = "global") -> Dict[str, Any]:
        seed_weight, mutator_weight, cost_weight = self.reward_weights
        reward = (
            seed_weight * seed_delta
            + mutator_weight * mutator_delta
            + self.dense_reward_weight * dense_reward
            - cost_weight * normalised_cost
        )

        stats = self.stats[action]
        stats.trials += trials
        stats.hard_successes += hard_successes
        stats.soft_score_sum += soft_score_sum
        next_state = self.state(next_seed_asr, action, context)
        next_max = max(
            (self.q_value(self.state(next_seed_asr, candidate, context), candidate)
             for candidate in self.actions),
            default=0.0,
        )
        old_value = self.q_value(state, action)
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value
        old_global = self.global_q[action]
        new_global = old_global + self.learning_rate * (reward - old_global)
        self.global_q[action] = new_global
        shared_updates = {}
        if self.similarity_credit > 0:
            source_tags = self.action_tags.get(action, set())
            for peer in self.actions:
                if peer == action:
                    continue
                peer_tags = self.action_tags.get(peer, set())
                union = source_tags | peer_tags
                similarity = len(source_tags & peer_tags) / len(union) if union else 0.0
                if similarity <= 0:
                    continue
                peer_state = self.state(next_seed_asr, peer, context)
                peer_old = self.q_value(peer_state, peer)
                peer_new = peer_old + self.learning_rate * self.similarity_credit * similarity * (
                    reward - peer_old)
                self.q_table[peer_state][peer] = peer_new
                peer_global_old = self.global_q[peer]
                self.global_q[peer] = peer_global_old + (
                    self.learning_rate * self.similarity_credit * similarity
                    * (reward - peer_global_old)
                )
                shared_updates[peer] = {
                    "similarity": similarity,
                    "old_q": peer_old,
                    "new_q": peer_new,
                    "old_global_q": peer_global_old,
                    "new_global_q": self.global_q[peer],
                }
        return {
            "reward": reward,
            "old_q": old_value,
            "new_q": new_value,
            "old_global_q": old_global,
            "new_global_q": new_global,
            "next_state": next_state,
            "epsilon": self.epsilon,
            "dense_reward": dense_reward,
            "shared_updates": shared_updates,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": self.actions,
            "epsilon": self.epsilon,
            "dense_reward_weight": self.dense_reward_weight,
            "similarity_credit": self.similarity_credit,
            "global_credit_weight": self.global_credit_weight,
            "action_tags": {name: sorted(tags) for name, tags in self.action_tags.items()},
            "q_table": dict(self.q_table),
            "global_q": self.global_q,
            "action_stats": {name: asdict(stats) | {"asr": stats.asr}
                             for name, stats in self.stats.items()},
        }
