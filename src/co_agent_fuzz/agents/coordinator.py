from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import asdict
from hashlib import sha256
from typing import Dict, Iterable, List, Optional, Sequence

from co_agent_fuzz.agents.critic import CriticAgent
from co_agent_fuzz.agents.filter import FilterAgent
from co_agent_fuzz.agents.helper import HelperAgent
from co_agent_fuzz.agents.mutator import MutatorAgent
from co_agent_fuzz.agents.strategy import StrategyAgent
from co_agent_fuzz.core.artifacts import ArtifactStore
from co_agent_fuzz.core.experience_memory import StrategyMemory
from co_agent_fuzz.core.message_bus import Message, MessageBus, MessageType
from co_agent_fuzz.core.oracle import CascadeOracle, build_l2_classifier
from co_agent_fuzz.core.rl_engine import QLearningEngine
from co_agent_fuzz.core.seed_pool import SeedPool
from co_agent_fuzz.core.surrogate import OnlineRandomForestSurrogate
from co_agent_fuzz.models import (
    CandidateRecord, DefenseEvaluation, FilterDecision, OracleDecision, RunStats,
)
from co_agent_fuzz.mutators import build_default_registry
from co_agent_fuzz.utils.model_client import ChatBackend
from co_agent_fuzz.utils.ppl_calculator import UnigramPerplexity


LOGGER = logging.getLogger("co_agent_fuzz")


class CoAgentCoordinator:
    """Coordinates Strategy Agent -> Attack Agent -> Critic Agent."""

    def __init__(self, config: Dict, seeds: Sequence[str], defenses: Sequence[Dict],
                 target_backend: ChatBackend, helper_backend: Optional[ChatBackend] = None,
                 filter_backend: Optional[ChatBackend] = None,
                 judge_backend: Optional[ChatBackend] = None):
        if not seeds:
            raise ValueError("At least one initial seed is required")
        if not defenses:
            raise ValueError("At least one defense is required")
        self.config = config
        self.defenses = list(defenses)
        self.target_backend = target_backend
        self.mode = str(config["task"]["mode"])
        self.verbose = bool(config["general"].get("verbose", True))
        random_seed = int(config["general"]["random_seed"])

        bus_config = config.get("message_bus", {})
        self.bus = MessageBus(int(bus_config.get("max_queue_size", 1000)))
        output = config["general"]
        self.artifacts = ArtifactStore(output["output_root"], output.get("run_name"))
        self.bus.subscribe("*", self.artifacts.event_handler)
        self.bus.set_delivery_observer(
            self.artifacts.delivery_handler, {MessageType.CRITIC_FEEDBACK})

        pool_config = config["seed_pool"]
        self.seed_pool = SeedPool(
            max_size=int(pool_config["max_size"]),
            ucb_exploration=float(pool_config.get("ucb_exploration", 1.4)),
            depth_penalty=float(pool_config.get("depth_penalty", 0.01)),
            feedback_exploitation_rate=float(
                pool_config.get("feedback_exploitation_rate", 0.40)),
            feedback_min_trials=int(pool_config.get("feedback_min_trials", 1)),
            random_seed=random_seed,
        )
        self.seed_pool.add_many(seeds, source="initial")

        registry = build_default_registry()
        enabled_names = config["mutators"].get("enabled", registry.names)
        self.registry = registry.select(enabled_names)
        collaboration = config.get("collaboration", {})

        rl_config = config["rl"]
        self.rl = QLearningEngine(
            self.registry.names,
            learning_rate=float(rl_config["learning_rate"]),
            discount_factor=float(rl_config["discount_factor"]),
            epsilon_start=float(rl_config["epsilon_start"]),
            epsilon_end=float(rl_config["epsilon_end"]),
            epsilon_decay=float(rl_config["epsilon_decay"]),
            bins=tuple(float(item) for item in rl_config["asr_bins"]),
            reward_weights=tuple(float(item) for item in rl_config["reward_weights"]),
            dense_reward_weight=float(rl_config.get("dense_reward_weight", 0.35)),
            similarity_credit=float(rl_config.get("similarity_credit", 0.20)),
            global_credit_weight=float(rl_config.get("global_credit_weight", 0.35)),
            action_tags=self.registry.action_tags,
            random_seed=random_seed,
        )

        surrogate_config = config["surrogate"]
        self.surrogate = OnlineRandomForestSurrogate(
            self.registry.names,
            enabled=bool(surrogate_config.get("enabled", True)),
            min_samples=int(surrogate_config.get("min_samples", 30)),
            retrain_interval=int(surrogate_config.get("retrain_interval", 10)),
            random_seed=random_seed,
        )
        self.attacker = MutatorAgent(
            self.registry,
            random_seed=random_seed,
            surrogate=self.surrogate,
            oversample_factor=int(collaboration.get("prompt_oversample_factor", 1)),
            feedback_ema_alpha=float(collaboration.get("feedback_ema_alpha", 0.30)),
        )
        self.mutator = self.attacker  # Backward-compatible attribute for external scripts.
        ppl = UnigramPerplexity().fit(item.text for item in self.seed_pool.records)
        self.filter = FilterAgent(config["filter"], self.bus, ppl, self.surrogate,
                                  local_backend=filter_backend)
        memory_config = collaboration.get("memory", {})
        memory_enabled = bool(collaboration.get("enabled", True)) and bool(
            memory_config.get("enabled", True))
        self.memory = StrategyMemory(
            enabled=memory_enabled,
            max_size=int(memory_config.get("max_size", 2000)),
            top_k=int(memory_config.get("top_k", 5)),
            min_similarity=float(memory_config.get("min_similarity", 0.05)),
            tier_weights={
                str(key): float(value)
                for key, value in memory_config.get("tier_weights", {}).items()
            } or None,
        )
        load_path = memory_config.get("load_path")
        if load_path:
            self.memory.load(load_path)
        context_document = "\n".join(
            [f"mode:{self.mode}", f"model:{config['target']['model']}"]
            + [
                f"defense:{item.get('pre_prompt', '')}\npost:{item.get('post_prompt', '')}"
                for item in self.defenses
            ]
        )
        context_hash = sha256(context_document.encode("utf-8")).hexdigest()[:12]
        context_prefix = f"{self.mode}|{config['target']['model']}|{context_hash}"
        strategy_config = {
            **collaboration,
            "surrogate_bias_weight": collaboration.get(
                "surrogate_bias_weight",
                config["surrogate"].get("action_bias_weight", 0.2)),
        }
        self.strategy = StrategyAgent(
            self.seed_pool, self.filter, self.rl, self.surrogate, self.memory,
            strategy_config, context_prefix, context_document,
        )
        self.critic = CriticAgent(self.memory, collaboration.get("critic", {}))
        self.bus.subscribe(MessageType.CRITIC_FEEDBACK, self.strategy.on_critic_feedback)
        self.bus.subscribe(MessageType.CRITIC_FEEDBACK, self.attacker.on_critic_feedback)
        helper_config = config["helper"]
        self.helper = HelperAgent(
            helper_backend, self.bus,
            enabled=bool(helper_config.get("enabled", False)),
            max_calls=int(helper_config.get("max_calls", 100)),
            max_tokens=int(helper_config.get("max_tokens", 256)),
            temperature=float(helper_config.get("temperature", 0.7)),
        )
        self.oracle = CascadeOracle(
            config["oracle"], build_l2_classifier(config["oracle"]["l2"]), judge_backend)
        self.stats = RunStats()
        self.first_success_query: Optional[int] = None
        self._iteration_asr: List[float] = []
        self._unique_success_hashes = set()
        self._started_at = 0.0

    async def run(self) -> Dict:
        self._started_at = time.perf_counter()
        await self.bus.start()
        self.artifacts.write_manifest(self.config, len(self.seed_pool), len(self.defenses))
        await self._phase("Pre-Jailbreak", 0, {
            "initial_seeds": len(self.seed_pool), "defenses": len(self.defenses)})
        self._say(0, "PRE-JAILBREAK",
                  f"normalised_seeds={len(self.seed_pool)} defenses={len(self.defenses)}")
        error: Optional[BaseException] = None
        try:
            while not self._should_stop():
                await self._run_iteration(self.stats.iterations + 1)
                interval = int(self.config["general"].get("checkpoint_interval", 10))
                if interval > 0 and self.stats.iterations % interval == 0:
                    await self._checkpoint()
                if self._converged():
                    self.stats.stop_reason = "asr_converged"
                    break
            if self.stats.stop_reason == "running":
                self.stats.stop_reason = self._budget_stop_reason()
        except BaseException as exc:
            error = exc
            self.stats.stop_reason = f"error:{type(exc).__name__}"
            LOGGER.exception("Co-AgentFuzz stopped unexpectedly")
        finally:
            self.stats.helper_calls = self.helper.calls
            self.stats.filter_model_calls = self.filter.local_model_calls
            self.stats.unique_successes = len(self._unique_success_hashes)
            self.stats.wall_seconds = time.perf_counter() - self._started_at
            await self.bus.publish(Message(
                MessageType.RUN_FINISHED, "coordinator", self._summary_payload(),
                self.stats.iterations))
            await self.bus.join()
            await self._checkpoint()
            await self.bus.close()
            self.artifacts.close()

        self._say(self.stats.iterations, "FINISHED",
                  f"reason={self.stats.stop_reason} ASR={self.stats.asr:.4f} "
                  f"success={self.stats.hard_successes}/{self.stats.target_queries} "
                  f"filtered={self.stats.variants_filtered}/{self.stats.variants_generated} "
                  f"output={self.artifacts.run_dir}")
        if error is not None:
            raise error
        return self._summary_payload()

    async def _run_iteration(self, iteration: int) -> None:
        self.stats.iterations = iteration
        await self._phase("Strategy-Agent", iteration)
        plan = self.strategy.plan(iteration, self.registry.names)
        seed, action, state = plan.seed, plan.action, plan.q_state
        seed_selection = self.seed_pool.last_selection_reason
        self.stats.feedback_seed_selections += int(seed_selection == "feedback_exploit")
        self.stats.evolved_seed_selections += int(seed.generation > 0)
        action_asr_before = self.rl.stats[action].asr
        seed_asr_before = seed.asr
        self.stats.strategy_retrieval_hits = self.strategy.retrieval_hits
        self._say(iteration, "STRATEGY",
                   f"seed={seed.seed_id} seedASR={seed.asr:.3f} mutator={action} "
                   f"epsilon={self.rl.epsilon:.3f} explore={plan.explored} "
                   f"seedPolicy={seed_selection} retrieved={len(plan.retrieved)} "
                   f"masked={len(plan.masked_actions)}")
        await self.bus.publish(Message(
            MessageType.SEED_SUBMIT, "strategy",
            {"seed_id": seed.seed_id, "seed_asr": seed.asr, "action": action,
             "seed_selection": seed_selection, "q_state": state,
             "explored": plan.explored}, iteration))
        await self.bus.publish(Message(
            MessageType.STRATEGY_RETRIEVED, "strategy",
            {"context_key": plan.context_key, "hits": len(plan.retrieved),
             "masked_actions": plan.masked_actions, "action_bias": plan.action_bias},
            iteration))

        await self._phase("Attack-Agent", iteration)
        peer_seeds = [item.text for item in self.seed_pool.records if item.seed_id != seed.seed_id]
        variants = self.attacker.generate(
            seed, action, int(self.config["general"]["variants_per_iteration"]),
            iteration, peer_seeds, self.mode)
        self.stats.variants_generated += len(variants)
        self.stats.prompt_proposals += self.attacker.last_proposal_count
        self._say(iteration, "ATTACK",
                   f"proposed={self.attacker.last_proposal_count} selected={len(variants)} "
                   f"lengths={[len(item.text) for item in variants]}")
        for variant in variants:
            await self.bus.publish(Message(
                MessageType.VARIANT_GENERATED, "attack",
                {"variant_id": variant.variant_id, "seed_id": seed.seed_id,
                 "mutator": action, "length": len(variant.text)}, iteration))

        await self._phase("Critic-Agent", iteration)
        candidates: List[CandidateRecord] = []
        query_start = self.stats.target_queries
        l3_start = self.stats.l3_calls
        for variant in variants:
            decision = await self.filter.evaluate(variant, seed)
            candidate = CandidateRecord(variant, decision)
            candidates.append(candidate)
            ppl_text = f"{decision.ppl:.1f}" if decision.ppl is not None else "NA"
            self._say(iteration, "FILTER",
                      f"variant={variant.variant_id} keep={decision.accepted} "
                      f"ppl={ppl_text} "
                      f"score={decision.score:.3f} reason={','.join(decision.reasons)}")
            if not decision.accepted:
                self.stats.variants_filtered += 1
                continue
            for defense_index, defense in enumerate(self.defenses):
                if self.stats.target_queries >= int(self.config["general"]["max_target_queries"]):
                    break
                evaluation = await self._evaluate_target(variant.text, defense, defense_index, iteration)
                candidate.evaluations.append(evaluation)
                self.stats.target_queries += 1
                self.stats.target_errors += int(evaluation.error is not None)
                self.stats.hard_successes += int(evaluation.oracle.benchmark_success)
                self.stats.oracle_successes += int(evaluation.oracle.hard_success)
                self.stats.suspicious_results += int(
                    evaluation.oracle.reason == "ambiguous_without_l3")
                self.stats.l3_calls += evaluation.oracle.l3_calls
                if evaluation.oracle.benchmark_success:
                    if self.first_success_query is None:
                        self.first_success_query = self.stats.target_queries
                    self._unique_success_hashes.add(
                        sha256(variant.text.encode("utf-8")).hexdigest())
                response_preview = " ".join(evaluation.response.split())[:80]
                self._say(iteration, "ORACLE",
                          f"variant={variant.variant_id} defense={evaluation.defense_id} "
                          f"strict={int(evaluation.oracle.benchmark_success)} "
                          f"layer={evaluation.oracle.layer} score={evaluation.oracle.score:.1f} "
                          f"response={response_preview!r}")

        feedback = self.critic.review(
            iteration=iteration,
            context_key=plan.context_key,
            context_text=plan.context_text,
            seed=seed,
            action=action,
            candidates=candidates,
        )
        batch_trials = sum(item.query_count for item in candidates)
        batch_successes = sum(item.hard_successes for item in candidates)
        batch_soft_score = sum(
            evaluation.oracle.score
            for candidate in candidates
            for evaluation in candidate.evaluations
        )
        self.stats.near_misses += feedback.outcome_counts.get("near_miss", 0)
        self.stats.semantic_rollbacks += feedback.rollbacks
        await self.bus.publish(Message(
            MessageType.CRITIC_FEEDBACK, "critic",
            {"action": action, "seed_id": seed.seed_id,
             "mean_reward": feedback.mean_reward,
             "dominant_outcome": feedback.dominant_outcome,
             "outcome_counts": feedback.outcome_counts,
             "memory_tiers": feedback.tiers,
             "trials": batch_trials,
             "hard_successes": batch_successes}, iteration))
        await self.bus.join()
        self.stats.feedback_cycles = min(
            self.strategy.feedback_updates, self.attacker.feedback_updates)
        if feedback.rollbacks:
            await self.bus.publish(Message(
                MessageType.ROLLBACK, "critic",
                {"count": feedback.rollbacks, "seed_id": seed.seed_id,
                 "reason": "semantic_deviation"}, iteration))
        self._say(iteration, "CRITIC",
                  f"outcomes={feedback.outcome_counts} dense={feedback.mean_reward:.3f} "
                  f"rollbacks={feedback.rollbacks} memory={self.memory.tier_counts}")

        expand_threshold = float(self.config["expand"]["min_score"])
        expandable = [item for item in candidates
                      if item.query_count
                      and not item.variant.metadata.get("rollback", False)
                      and item.variant.metadata.get("memory_tier") in {"effective", "promising"}
                      and (item.hard_successes or item.mean_score >= expand_threshold)]
        added_count = 0
        for candidate in expandable:
            new_seed, added = self.seed_pool.add(
                candidate.variant.text,
                parent_id=seed.seed_id,
                source="mutator",
                generation=candidate.variant.generation,
                iteration=iteration,
            )
            if new_seed.seed_id != seed.seed_id:
                self.seed_pool.update(
                    new_seed.seed_id,
                    candidate.query_count,
                    candidate.hard_successes,
                    sum(item.oracle.score for item in candidate.evaluations),
                )
            if added:
                candidate.expanded_seed_ids.append(new_seed.seed_id)
                added_count += 1
                await self.bus.publish(Message(
                    MessageType.SEED_EXPANDED, "coordinator",
                    {"seed_id": new_seed.seed_id, "parent_id": seed.seed_id,
                     "source": "mutator"}, iteration))

        helper_before = self.helper.calls
        if expandable:
            best = max(expandable, key=lambda item: (item.hard_successes, item.mean_score))
            for enhanced in await self.helper.enhance(best.variant.text, self.mode, iteration):
                helper_seed, added = self.seed_pool.add(
                    enhanced, parent_id=best.variant.seed_id, source="helper",
                    generation=best.variant.generation + 1, iteration=iteration)
                if added:
                    best.expanded_seed_ids.append(helper_seed.seed_id)
                    added_count += 1

        self.seed_pool.update(seed.seed_id, batch_trials, batch_successes, batch_soft_score)
        predicted_action_asr = (
            (self.rl.stats[action].hard_successes + batch_successes)
            / max(1, self.rl.stats[action].trials + batch_trials))
        query_cost = self.stats.target_queries - query_start
        l3_cost = self.stats.l3_calls - l3_start
        helper_cost = self.helper.calls - helper_before
        weights = self.config["cost"]
        normalised_cost = (
            query_cost * float(weights["target_query"])
            + l3_cost * float(weights["l3_call"])
            + helper_cost * float(weights["helper_call"])
        ) / max(1.0, batch_trials)
        q_update = self.strategy.learn(
            plan,
            next_seed_asr=seed.asr,
            seed_delta=seed.asr - seed_asr_before,
            mutator_delta=predicted_action_asr - action_asr_before,
            normalised_cost=normalised_cost,
            trials=batch_trials,
            hard_successes=batch_successes,
            soft_score_sum=batch_soft_score,
            dense_reward=feedback.mean_reward,
        )
        self.filter.report_batch(action, batch_successes, iteration)
        for candidate in candidates:
            if candidate.query_count:
                self.surrogate.observe(
                    candidate.variant.text, seed, action, candidate.hard_successes > 0)
            self.artifacts.write_candidate(candidate)
        removed = []
        prune_interval = int(self.config["seed_pool"].get("pruning_interval", 100))
        if prune_interval > 0 and iteration % prune_interval == 0:
            removed = self.seed_pool.prune(float(self.config["seed_pool"]["min_asr_for_keep"]))

        await self.bus.publish(Message(
            MessageType.Q_UPDATE, "strategy",
            {"state": state, "action": action, **q_update}, iteration))
        await self.bus.join()
        iteration_queries = self.stats.target_queries - query_start
        iteration_successes = sum(item.hard_successes for item in candidates)
        iteration_asr = iteration_successes / iteration_queries if iteration_queries else 0.0
        self._iteration_asr.append(iteration_asr)
        metric = {
            "iteration": iteration,
            "seed_id": seed.seed_id,
            "seed_selection": seed_selection,
            "mutator": action,
            "prompt_proposals": self.attacker.last_proposal_count,
            "variants": len(candidates),
            "filtered": sum(not item.filter.accepted for item in candidates),
            "queries": iteration_queries,
            "hard_successes": iteration_successes,
            "iteration_asr": iteration_asr,
            "cumulative_asr": self.stats.asr,
            "new_seeds": added_count,
            "pruned_seeds": len(removed),
            "q_reward": q_update["reward"],
            "q_value": q_update["new_q"],
            "global_q_value": q_update["new_global_q"],
            "dense_reward": feedback.mean_reward,
            "critic_outcomes": feedback.outcome_counts,
            "semantic_rollbacks": feedback.rollbacks,
            "memory_hits": len(plan.retrieved),
            "shared_q_updates": len(q_update["shared_updates"]),
            "epsilon": q_update["epsilon"],
            "surrogate_ready": self.surrogate.ready,
            "feedback_cycles": self.stats.feedback_cycles,
            "target_queries_total": self.stats.target_queries,
        }
        self.artifacts.write_metric(metric)
        self._say(iteration, "LEARN",
                  f"new_seeds={added_count} pool={len(self.seed_pool)} reward={q_update['reward']:.3f} "
                  f"Q={q_update['new_q']:.3f} iterationASR={iteration_asr:.3f} "
                  f"cumulativeASR={self.stats.asr:.3f} queries={self.stats.target_queries}")

    async def _evaluate_target(self, attack: str, defense: Dict,
                               defense_index: int, iteration: int) -> DefenseEvaluation:
        defense_id = str(defense.get("sample_id", defense_index))
        messages = [{"role": "system", "content": str(defense.get("pre_prompt", ""))},
                    {"role": "user", "content": attack}]
        post_prompt = str(defense.get("post_prompt", ""))
        if post_prompt:
            messages.append({"role": "user", "content": post_prompt})
        started = time.perf_counter()
        try:
            response = await self.target_backend.complete(
                messages,
                max_tokens=int(self.config["target"]["max_tokens"]),
                temperature=float(self.config["target"]["temperature"]),
            )
            latency = time.perf_counter() - started
            await self.bus.publish(Message(
                MessageType.TARGET_RESPONSE, "target",
                {"defense_id": defense_id, "response_length": len(response),
                 "latency_seconds": latency}, iteration))
            oracle = await self.oracle.evaluate(attack, response, defense, self.mode)
            await self.bus.publish(Message(
                MessageType.ORACLE_RESULT, "oracle",
                {"defense_id": defense_id, **asdict(oracle)}, iteration))
            return DefenseEvaluation(defense_id, response, oracle, latency)
        except Exception as exc:
            latency = time.perf_counter() - started
            error = f"{type(exc).__name__}: {exc}"
            oracle = OracleDecision(0.0, False, "ERROR", 1.0, error,
                                    benchmark_success=False)
            return DefenseEvaluation(defense_id, "", oracle, latency, error=error)

    async def _phase(self, phase: str, iteration: int,
                     extra: Optional[Dict] = None) -> None:
        await self.bus.publish(Message(
            MessageType.PHASE_STARTED, "coordinator",
            {"phase": phase, **(extra or {})}, iteration))

    async def _checkpoint(self) -> None:
        self.artifacts.checkpoint(
            self.stats,
            self.rl.to_dict(),
            self.seed_pool.to_dict(),
            self.memory.to_dict(),
        )
        await self.bus.publish(Message(
            MessageType.CHECKPOINT, "coordinator",
            {"iterations": self.stats.iterations, "queries": self.stats.target_queries},
            self.stats.iterations))
        await self.bus.join()

    def _should_stop(self) -> bool:
        general = self.config["general"]
        return (self.stats.iterations >= int(general["max_iterations"])
                or self.stats.target_queries >= int(general["max_target_queries"]))

    def _budget_stop_reason(self) -> str:
        general = self.config["general"]
        if self.stats.target_queries >= int(general["max_target_queries"]):
            return "target_query_budget"
        return "iteration_budget"

    def _converged(self) -> bool:
        general = self.config["general"]
        min_iterations = int(general.get("convergence_min_iterations", 50))
        target_asr = float(general.get("convergence_asr_threshold", 1.1))
        if self.stats.iterations >= min_iterations and self.stats.asr >= target_asr:
            return True
        window = int(general.get("convergence_window", 20))
        if window <= 0 or len(self._iteration_asr) < 2 * window:
            return False
        previous = statistics.fmean(self._iteration_asr[-2 * window:-window])
        current = statistics.fmean(self._iteration_asr[-window:])
        return current - previous <= float(general.get("convergence_min_delta", 0.005))

    def _summary_payload(self) -> Dict:
        return {
            **self.stats.to_dict(),
            "first_success_query": self.first_success_query,
            "seed_pool_size": len(self.seed_pool),
            "experience_memory_size": len(self.memory.records),
            "experience_memory_tiers": self.memory.tier_counts,
            "pipeline": ["strategy_agent", "attack_agent", "critic_agent"],
            "message_bus_errors": self.bus.errors,
            "output_dir": str(self.artifacts.run_dir),
            "total_llm_calls": (
                self.stats.target_queries + self.stats.helper_calls + self.stats.l3_calls
                + self.stats.filter_model_calls),
        }

    def _say(self, iteration: int, phase: str, message: str) -> None:
        if self.verbose:
            LOGGER.info("[%04d %-12s] %s", iteration, phase, message)
