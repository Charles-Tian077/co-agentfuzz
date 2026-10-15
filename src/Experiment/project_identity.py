"""Normalize display labels when reading immutable historical results."""
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def identity():
    return json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def canonical_method(name):
    metadata = identity()
    aliases = {alias.casefold() for alias in metadata['method_aliases']}
    if isinstance(name, str) and name.casefold() in aliases:
        return metadata['display_name']
    return name
