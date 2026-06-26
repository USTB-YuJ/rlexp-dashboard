from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class RemoteSource:
    name: str
    host: str
    user: str
    port: int
    remote_log_root: str
    project: str
    method: str = "rsync"
    include_patterns: Sequence[str] = field(default_factory=tuple)
    exclude_patterns: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class SyncPlan:
    source: RemoteSource
    method: str
    local_path: Path
    command: List[str]
    dry_run: bool
    include_videos: bool


def build_sync_plan(
    source: RemoteSource,
    cache_root: Path,
    dry_run: bool = True,
    include_videos: bool = False,
) -> SyncPlan:
    method = source.method.lower()
    local_path = Path(cache_root) / source.name / source.project / "logs" / "rsl_rl"

    if method == "rsync":
        command = _build_rsync_command(source, local_path, dry_run=dry_run, include_videos=include_videos)
    elif method == "scp":
        command = _build_scp_command(source, local_path)
    elif method in {"ssh-tar", "tar"}:
        command = _build_ssh_tar_command(source, local_path)
        method = "ssh-tar"
    else:
        raise ValueError(f"Unsupported sync method: {source.method}")

    return SyncPlan(
        source=source,
        method=method,
        local_path=local_path,
        command=command,
        dry_run=dry_run,
        include_videos=include_videos,
    )


def _build_rsync_command(
    source: RemoteSource,
    local_path: Path,
    dry_run: bool,
    include_videos: bool,
) -> List[str]:
    command = ["rsync", "-az", "--prune-empty-dirs"]
    if dry_run:
        command.append("--dry-run")
    command.extend(["-e", f"ssh -p {source.port}"])
    for pattern in _default_include_patterns(include_videos) + list(source.include_patterns):
        command.append(f"--include={pattern}")
    for pattern in _default_exclude_patterns(include_videos) + list(source.exclude_patterns):
        command.append(f"--exclude={pattern}")
    command.extend(
        [
            _remote_location(source, trailing_slash=True),
            _local_location(local_path),
        ]
    )
    return command


def _build_scp_command(source: RemoteSource, local_path: Path) -> List[str]:
    return [
        "scp",
        "-P",
        str(source.port),
        "-r",
        _remote_location(source, trailing_slash=False),
        str(local_path),
    ]


def _build_ssh_tar_command(source: RemoteSource, local_path: Path) -> List[str]:
    remote = f"{source.user}@{source.host}"
    return [
        "ssh",
        "-p",
        str(source.port),
        remote,
        f"tar -C {source.remote_log_root} -cf - .",
        "|",
        "tar",
        "-C",
        str(local_path),
        "-xf",
        "-",
    ]


def _default_include_patterns(include_videos: bool) -> List[str]:
    patterns = [
        "*/",
        "params/***",
        "events.out.tfevents*",
        "model_*.pt",
        "config.*",
    ]
    if include_videos:
        patterns.append("videos/***")
    return patterns


def _default_exclude_patterns(include_videos: bool) -> List[str]:
    patterns = []
    if not include_videos:
        patterns.append("videos/***")
    patterns.append("*")
    return patterns


def _remote_location(source: RemoteSource, trailing_slash: bool) -> str:
    root = source.remote_log_root.rstrip("/")
    suffix = "/" if trailing_slash else ""
    return f"{source.user}@{source.host}:{root}{suffix}"


def _local_location(local_path: Path) -> str:
    return f"{local_path}/"
