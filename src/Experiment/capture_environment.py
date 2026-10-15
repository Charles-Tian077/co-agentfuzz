from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from Experiment.layout import repository_root, dataset_root, output_root, default_config, resolve_legacy_path
REPO_ROOT = repository_root()
PACKAGES = ("openai", "httpx", "pandas", "scikit-learn", "tiktoken", "PyYAML", "pytest")
TRACKED_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "src/co_agent_fuzz/config/default.yaml",
    "src/Experiment/compare_fuzzers.py",
    "src/Experiment/compare_paper_metrics.py",
    "src/Experiment/run_validation_matrix.py",
    "src/Experiment/run_ablation_matrix.py",
    "src/Experiment/tune_mutators.py",
    "src/coagent_repro/replicate_records.py",
    "docs/REPRODUCIBILITY.md",
    "experiments/slurm/slurm_records.sbatch",
    "datasets/synthetic/manifest.json",
    "models/registry.json",
)


def command_output(command: List[str]) -> Dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "output": ""}
    try:
        completed = subprocess.run(
            [executable, *command[1:]], cwd=REPO_ROOT, text=True,
            encoding="utf-8", errors="replace", capture_output=True,
            timeout=20, check=False,
        )
        return {
            "available": True,
            "command": command,
            "returncode": completed.returncode,
            "output": (completed.stdout + completed.stderr).strip(),
        }
    except subprocess.TimeoutExpired:
        return {"available": True, "command": command, "returncode": None,
                "output": "TIMEOUT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a non-secret experiment environment manifest")
    parser.add_argument("--label", default=platform.node())
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hashes = {}
    for relative in TRACKED_FILES:
        path = REPO_ROOT / relative
        hashes[relative] = sha256(path) if path.is_file() else None
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "file_sha256": hashes,
        "ollama_version": command_output(["ollama", "--version"]),
        "ollama_models": command_output(["ollama", "list"]),
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "git": command_output(["git", "rev-parse", "HEAD"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
