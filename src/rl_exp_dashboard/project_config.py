from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .structured_loader import load_structured_file
from .sync import RemoteSource


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    local_cache_root: Path
    parser_profile: str = "generic_tensorboard"
    preferred_metrics: Tuple[str, ...] = field(default_factory=tuple)
    log_patterns: Tuple[str, ...] = field(default_factory=tuple)
    tag_schema: Tuple[str, ...] = field(default_factory=tuple)
    remote_sources: Tuple[RemoteSource, ...] = field(default_factory=tuple)


def load_project_config(path: Path) -> ProjectConfig:
    data = load_structured_file(Path(path).expanduser())
    return project_config_from_dict(data)


def project_config_from_dict(data: Dict[str, Any]) -> ProjectConfig:
    name = _required_string(data, "name")
    cache_root = Path(str(data.get("local_cache_root") or "~/rl-exp-dashboard/cache")).expanduser()
    parser_profile = str(data.get("parser_profile") or "generic_tensorboard")
    remote_sources = tuple(_remote_source_from_dict(name, item) for item in _remote_source_items(data.get("remote_sources")))
    return ProjectConfig(
        name=name,
        local_cache_root=cache_root,
        parser_profile=parser_profile,
        preferred_metrics=_string_tuple(data.get("preferred_metrics")),
        log_patterns=_string_tuple(data.get("log_patterns")),
        tag_schema=_string_tuple(data.get("tag_schema")),
        remote_sources=remote_sources,
    )


def _required_string(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Project config requires `{key}`.")
    return str(value)


def _remote_source_items(value: Any) -> Sequence[Dict[str, Any]]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return [dict({"name": name}, **details) for name, details in value.items() if isinstance(details, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise ValueError("Project config `remote_sources` must be a list or mapping.")


def _remote_source_from_dict(project: str, data: Dict[str, Any]) -> RemoteSource:
    return RemoteSource(
        name=_required_string(data, "name"),
        host=_required_string(data, "host"),
        user=_required_string(data, "user"),
        port=int(data.get("port", 22)),
        remote_log_root=_required_string(data, "remote_log_root"),
        project=project,
        method=str(data.get("method") or "rsync"),
        include_patterns=_string_tuple(data.get("include_patterns")),
        exclude_patterns=_string_tuple(data.get("exclude_patterns")),
    )


def _string_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)
