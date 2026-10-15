"""Portable release paths; no credentials, model calls or controller changes."""
import os
from pathlib import Path


def repository_root():
    configured = os.environ.get('COAGENT_PROJECT_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'src').is_dir():
            return parent
    # Installed wheels have no checkout; explicit paths or the caller's cwd apply.
    return Path.cwd().resolve()


def dataset_root():
    value = os.environ.get('COAGENT_DATA_ROOT')
    return Path(value).expanduser().resolve() if value else repository_root() / 'datasets' / 'private'


def output_root():
    value = os.environ.get('COAGENT_OUTPUT_ROOT')
    return Path(value).expanduser().resolve() if value else repository_root() / 'experiments' / 'outputs'


def default_config():
    return Path(__file__).resolve().parents[1] / 'co_agent_fuzz' / 'config' / 'default.yaml'


def resolve_legacy_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'Datasets':
        return dataset_root().joinpath(*path.parts[1:])
    if path.parts and path.parts[0] == 'Results':
        return output_root().joinpath(*path.parts[1:])
    return repository_root() / path
