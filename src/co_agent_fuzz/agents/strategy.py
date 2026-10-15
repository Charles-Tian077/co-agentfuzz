from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List

from co_agent_fuzz.agents.filter import FilterAgent
from co_agent_fuzz.core.experience_memory import RetrievedExperience, StrategyMemory
from co_agent_fuzz.core.message_bus import Message
from co_agent_fuzz.core.rl_engine import QLearningEngine
from co_agent_fuzz.core.seed_pool import SeedPool
from co_agent_fuzz.core.surrogate import OnlineRandomForestSurrogate
from co_agent_fuzz.models import SeedRecord


@dataclass(frozen=True)
class StrategyPlan:
    seed: SeedRecord
    action: str
    q_state: str
    context_key: str
    context_text: str
    explored: bool
    active_actions: int
    masked_actions: List[str]
    action_bias: Dict[str, float]
    retrieved: List[RetrievedExperience]


class StrategyAgent:
    """Stage 1: retrieve experience and choose a non-repeating search action."""

    def __init__(self, seed_pool: SeedPool, filter_agent: FilterAgent,
                 rl: QLearningEngine, surrogate: OnlineRandomForestSurrogate,
                 memory: StrategyMemory, config: Dict, context_prefix: str,
                 context_document: str):
        self.seed_pool = seed_pool
        self.filter = filter_agent
        self.rl = rl
        self.surrogate = surrogate
        self.memory = memory
        self.config = config
        self.context_prefix = context_prefix
        self.context_document = context_document
        self.last_outcome = "start"
        self.retrieval_hits = 0
        self.feedback_updates = 0
        self._history: Dict[str, Deque[str]] = defaultdict(deque)

    def plan(self, iteration: int, all_actions: Iterable[str]) -> StrategyPlan:
        seed = self.seed_pool.select_ucb(iteration)
        active = self.filter.active_mutators(all_actions, iteration)
        context_key = f"{self.context_prefix}|{self.last_outcome}"
        context_text = (
            f"{self.context_document}\nlast_outcome:{self.last_outcome}\n"
            f"seed:{seed.text}"
        )
        memory_bias, retrieved = self.memory.retrieve(context_text, active)
        self.retrieval_hits += len(retrieved)

        window = max(0, int(self.config.get("no_repeat_window", 3)))
        history = self._history[self.context_prefix]
        recent = set(history)
        candidates = [action for action in active if action not in recent] or list(active)
        masked = [action for action in active if action not in candidates]

        surrogate_weight = float(self.config.get("surrogate_bias_weight", 0.2))
        action_bias = self.surrogate.action_bias(seed, surrogate_weight)
        memory_weight = float(self.config.get("memory_bias_weight", 0.35))
        for action, value in memory_bias.items():
            action_bias[action] = action_bias.get(action, 0.0) + memory_weight * value

        action, state, explored = self.rl.select(
            seed.asr,
            candidates,
            action_bias,
            context=context_key,
            selection_policy=str(self.config.get("selection_policy", "softmax")),
            temperature=float(self.config.get("temperature", 0.15)),
            q_smoothing=float(self.config.get("q_smoothing", 0.01)),
        )
        if window:
            history.append(action)
            while len(history) > window:
                history.popleft()
        return StrategyPlan(
            seed=seed,
            action=action,
            q_state=state,
            context_key=context_key,
            context_text=context_text,
            explored=explored,
            active_actions=len(active),
            masked_actions=masked,
            action_bias=action_bias,
            retrieved=retrieved,
        )

    def observe(self, outcome: str) -> None:
        self.last_outcome = outcome

    def on_critic_feedback(self, message: Message) -> None:
        """Consume Critic feedback before the next strategy decision."""
        self.observe(str(message.payload.get("dominant_outcome", "empty")))
        self.feedback_updates += 1

    def learn(self, plan: StrategyPlan, *, next_seed_asr: float,
              seed_delta: float, mutator_delta: float,
              normalised_cost: float, trials: int, hard_successes: int,
              soft_score_sum: float, dense_reward: float) -> Dict[str, Any]:
        """Apply the Critic reward to the Strategy Agent's policy."""
        return self.rl.update(
            plan.q_state,
            plan.action,
            next_seed_asr,
            seed_delta,
            mutator_delta,
            normalised_cost,
            trials,
            hard_successes,
            soft_score_sum,
            dense_reward=dense_reward,
            context=plan.context_key,
        )
