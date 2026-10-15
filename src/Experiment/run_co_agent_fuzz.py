from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")

from co_agent_fuzz.config import load_config
from co_agent_fuzz.data import load_experiment_data
from co_agent_fuzz.utils.model_client import OpenAIChatBackend


def _backend(section: dict, fallback: dict | None = None) -> OpenAIChatBackend:
    merged = {**(fallback or {}), **section}
    api_key = os.environ.get(str(merged.get("api_key_env", ""))) or str(merged.get("api_key", "ollama"))
    return OpenAIChatBackend(
        model=str(merged["model"]),
        api_key=api_key,
        base_url=merged.get("base_url") or None,
        timeout_seconds=float(merged.get("timeout_seconds", 120)),
        max_concurrency=int(merged.get("max_concurrency", 1)),
    )


async def run(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.set)
    if args.model:
        config["target"]["model"] = args.model
    if args.api_base:
        config["target"]["base_url"] = args.api_base
    if args.max_queries:
        config["general"]["max_target_queries"] = args.max_queries
    if args.max_iterations:
        config["general"]["max_iterations"] = args.max_iterations
    if args.run_name:
        config["general"]["run_name"] = args.run_name

    def resolve(path: str) -> Path:
        return resolve_legacy_path(path)

    task = config["task"]
    seeds, defenses = load_experiment_data(
        resolve(task["seeds_path"]), resolve(task["defenses_path"]),
        defense_index=int(task.get("defense_index", 0)),
        all_defenses=bool(task.get("all_defenses", False)),
        max_defenses=int(task.get("max_defenses", 0)),
    )
    output_root = Path(config["general"]["output_root"])
    if not output_root.is_absolute():
        config["general"]["output_root"] = str(resolve_legacy_path(output_root))

    target = _backend(config["target"])
    helper = _backend(config["helper"], config["target"]) if config["helper"].get("enabled", False) else None
    local_filter = _backend(config["filter"]["local_model"], config["target"]) if config["filter"]["local_model"].get("enabled", False) else None
    judge = _backend(config["oracle"]["l3"], config["target"]) if config["oracle"]["l3"].get("enabled", False) else None

    from co_agent_fuzz.agents.coordinator import CoAgentCoordinator
    coordinator = CoAgentCoordinator(config, seeds, defenses, target, helper, local_filter, judge)
    try:
        summary = await coordinator.run()
        usages = {
            name: backend.usage() for name, backend in (
                ("target", target), ("helper", helper),
                ("filter", local_filter), ("judge", judge)) if backend is not None
        }
        summary["backend_usage"] = usages
        summary["prompt_tokens"] = sum(item["prompt_tokens"] for item in usages.values())
        summary["completion_tokens"] = sum(item["completion_tokens"] for item in usages.values())
        summary["total_tokens"] = sum(item["total_tokens"] for item in usages.values())
        return summary
    finally:
        for backend in {id(item): item for item in (target, helper, local_filter, judge) if item}.values():
            backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Co-AgentFuzz")
    parser.add_argument("--config", default=str(default_config()))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a YAML value; repeat for multiple values")
    parser.add_argument("--model", help="Override target model")
    parser.add_argument("--api-base", help="Override target OpenAI-compatible base URL")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    summary = asyncio.run(run(args))
    print("CO_AGENT_FUZZ_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
