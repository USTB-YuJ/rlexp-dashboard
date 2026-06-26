from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .models import ConfigDiff


FlatConfig = Dict[str, Any]


def flatten_config(config: Any, prefix: str = "") -> FlatConfig:
    """Flatten nested dictionaries/lists into dotted paths."""

    if isinstance(config, dict):
        flattened: FlatConfig = {}
        for key, value in config.items():
            path = _join_path(prefix, str(key))
            flattened.update(flatten_config(value, path))
        return flattened

    if isinstance(config, (list, tuple)):
        flattened = {}
        for index, value in enumerate(config):
            path = _join_path(prefix, str(index))
            flattened.update(flatten_config(value, path))
        return flattened

    return {prefix: config}


def diff_configs(before: Any, after: Any) -> List[ConfigDiff]:
    before_flat = flatten_config(before)
    after_flat = flatten_config(after)
    paths = sorted(set(before_flat) | set(after_flat))
    diffs: List[ConfigDiff] = []

    for path in paths:
        has_before = path in before_flat
        has_after = path in after_flat
        before_value = before_flat.get(path)
        after_value = after_flat.get(path)

        if not has_before:
            diffs.append(ConfigDiff(path=path, kind="added", after=after_value))
        elif not has_after:
            diffs.append(ConfigDiff(path=path, kind="removed", before=before_value))
        elif type(before_value) is not type(after_value):
            diffs.append(
                ConfigDiff(path=path, kind="type_changed", before=before_value, after=after_value)
            )
        elif before_value != after_value:
            diffs.append(ConfigDiff(path=path, kind="changed", before=before_value, after=after_value))

    return diffs


def _join_path(prefix: str, part: str) -> str:
    if not prefix:
        return part
    return f"{prefix}.{part}"
