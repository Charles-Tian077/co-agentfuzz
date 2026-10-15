from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{sha256(payload.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class SeedRecord:
    seed_id: str
    text: str
    parent_id: Optional[str] = None
    source: str = "initial"
    generation: int = 0
    created_iteration: int = 0
    last_selected_iteration: int = 0
    selections: int = 0
    trials: int = 0
    hard_successes: int = 0
    soft_score_sum: float = 0.0

    @property
    def asr(self) -> float:
        return self.hard_successes / self.trials if self.trials else 0.0

    @property
    def mean_score(self) -> float:
        return self.soft_score_sum / self.trials if self.trials else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(asr=self.asr, mean_score=self.mean_score)
        return data


@dataclass
class VariantRecord:
    variant_id: str
    seed_id: str
    text: str
    mutator: str
    generation: int
    iteration: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterDecision:
    accepted: bool
    score: float
    ppl: Optional[float]
    reasons: List[str] = field(default_factory=list)
    local_model_score: Optional[float] = None
    surrogate_score: Optional[float] = None


@dataclass
class OracleDecision:
    score: float
    hard_success: bool
    layer: str
    confidence: float
    reason: str
    l1_score: Optional[float] = None
    l2_score: Optional[float] = None
    l3_score: Optional[float] = None
    l3_calls: int = 0
    benchmark_success: bool = False


@dataclass
class DefenseEvaluation:
    defense_id: str
    response: str
    oracle: OracleDecision
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class CandidateRecord:
    variant: VariantRecord
    filter: FilterDecision
    evaluations: List[DefenseEvaluation] = field(default_factory=list)
    expanded_seed_ids: List[str] = field(default_factory=list)

    @property
    def query_count(self) -> int:
        return len(self.evaluations)

    @property
    def hard_successes(self) -> int:
        return sum(int(item.oracle.benchmark_success) for item in self.evaluations)

    @property
    def oracle_successes(self) -> int:
        return sum(int(item.oracle.hard_success) for item in self.evaluations)

    @property
    def asr(self) -> float:
        return self.hard_successes / self.query_count if self.query_count else 0.0

    @property
    def mean_score(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(item.oracle.score for item in self.evaluations) / self.query_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": asdict(self.variant),
            "filter": asdict(self.filter),
            "evaluations": [asdict(item) for item in self.evaluations],
            "expanded_seed_ids": self.expanded_seed_ids,
            "query_count": self.query_count,
            "hard_successes": self.hard_successes,
            "oracle_successes": self.oracle_successes,
            "asr": self.asr,
            "mean_score": self.mean_score,
        }


@dataclass
class RunStats:
    iterations: int = 0
    variants_generated: int = 0
    variants_filtered: int = 0
    target_queries: int = 0
    target_errors: int = 0
    hard_successes: int = 0
    oracle_successes: int = 0
    suspicious_results: int = 0
    l3_calls: int = 0
    helper_calls: int = 0
    filter_model_calls: int = 0
    unique_successes: int = 0
    near_misses: int = 0
    semantic_rollbacks: int = 0
    strategy_retrieval_hits: int = 0
    feedback_cycles: int = 0
    feedback_seed_selections: int = 0
    evolved_seed_selections: int = 0
    prompt_proposals: int = 0
    stop_reason: str = "running"
    wall_seconds: float = 0.0

    @property
    def asr(self) -> float:
        return self.hard_successes / self.target_queries if self.target_queries else 0.0

    @property
    def oracle_asr(self) -> float:
        return self.oracle_successes / self.target_queries if self.target_queries else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["asr"] = self.asr
        data["oracle_asr"] = self.oracle_asr
        return data
