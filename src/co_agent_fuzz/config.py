from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        return "" if default is None else default

    return _ENV_PATTERN.sub(replace, value)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_config_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def parse_overrides(items: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        key, raw = item.split("=", 1)
        set_config_value(result, key, yaml.safe_load(raw))
    return result


def _validate(config: Dict[str, Any]) -> None:
    general = config["general"]
    if general["max_iterations"] <= 0:
        raise ValueError("general.max_iterations must be positive")
    if general["variants_per_iteration"] <= 0:
        raise ValueError("general.variants_per_iteration must be positive")
    if general["max_target_queries"] <= 0:
        raise ValueError("general.max_target_queries must be positive")

    rl = config["rl"]
    for name in ("epsilon_start", "epsilon_end", "learning_rate", "discount_factor"):
        if not 0.0 <= float(rl[name]) <= 1.0:
            raise ValueError(f"rl.{name} must be in [0, 1]")
    if not 0.0 <= float(rl.get("global_credit_weight", 0.35)) <= 1.0:
        raise ValueError("rl.global_credit_weight must be in [0, 1]")
    if rl["epsilon_end"] > rl["epsilon_start"]:
        raise ValueError("rl.epsilon_end cannot exceed rl.epsilon_start")

    collaboration = config.get("collaboration", {})
    if collaboration:
        if int(collaboration.get("no_repeat_window", 0)) < 0:
            raise ValueError("collaboration.no_repeat_window cannot be negative")
        if int(collaboration.get("prompt_oversample_factor", 1)) < 1:
            raise ValueError("collaboration.prompt_oversample_factor must be positive")
        feedback_alpha = float(collaboration.get("feedback_ema_alpha", 0.30))
        if not 0 < feedback_alpha <= 1:
            raise ValueError("collaboration.feedback_ema_alpha must be in (0, 1]")
        temperature = float(collaboration.get("temperature", 0.15))
        if temperature <= 0:
            raise ValueError("collaboration.temperature must be positive")
        semantic_threshold = float(
            collaboration.get("critic", {}).get("semantic_similarity_threshold", 0.35))
        if not 0 <= semantic_threshold <= 1:
            raise ValueError("collaboration.critic.semantic_similarity_threshold must be in [0, 1]")

    seed_pool = config["seed_pool"]
    feedback_rate = float(seed_pool.get("feedback_exploitation_rate", 0.40))
    if not 0 <= feedback_rate <= 1:
        raise ValueError("seed_pool.feedback_exploitation_rate must be in [0, 1]")
    if int(seed_pool.get("feedback_min_trials", 1)) < 1:
        raise ValueError("seed_pool.feedback_min_trials must be positive")

    oracle = config["oracle"]
    if not 0 <= oracle["l1"]["negative_threshold"] < oracle["l1"]["positive_threshold"] <= 1:
        raise ValueError("oracle.l1 thresholds must satisfy 0 <= negative < positive <= 1")
    if not 0 <= oracle["l2"]["negative_threshold"] < oracle["l2"]["positive_threshold"] <= 1:
        raise ValueError("oracle.l2 thresholds must satisfy 0 <= negative < positive <= 1")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    override_map = parse_overrides(overrides)
    config = _expand_env(_deep_merge(loaded, override_map))
    config["_meta"] = {"config_path": str(config_path)}
    _validate(config)
    return config
