from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence


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
    include_checkpoints: bool


@dataclass(frozen=True)
class SyncExecutionResult:
    status: str
    return_code: int
    stdout: str
    stderr: str
    command: List[str]


SyncRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_sync_plan(
    source: RemoteSource,
    cache_root: Path,
    dry_run: bool = True,
    include_videos: bool = False,
    include_checkpoints: bool = False,
) -> SyncPlan:
    method = source.method.lower()
    local_path = Path(cache_root) / source.name / source.project / "logs" / "rsl_rl"

    if method == "rsync":
        command = _build_rsync_command(
            source,
            local_path,
            dry_run=dry_run,
            include_videos=include_videos,
            include_checkpoints=include_checkpoints,
        )
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
        include_checkpoints=include_checkpoints,
    )


def execute_sync_plan(
    plan: SyncPlan,
    runner: SyncRunner | None = None,
) -> SyncExecutionResult:
    if plan.dry_run:
        raise ValueError("Cannot execute a dry-run sync plan")
    if plan.method == "ssh-tar":
        raise ValueError("ssh-tar execution is not implemented; use rsync or scp")

    plan.local_path.mkdir(parents=True, exist_ok=True)
    runner = runner or subprocess.run
    completed = runner(
        plan.command,
        capture_output=True,
        text=True,
        check=False,
    )
    return SyncExecutionResult(
        status="completed" if completed.returncode == 0 else "failed",
        return_code=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=plan.command,
    )


def _build_rsync_command(
    source: RemoteSource,
    local_path: Path,
    dry_run: bool,
    include_videos: bool,
    include_checkpoints: bool,
) -> List[str]:
    command = ["rsync", "-az", "--prune-empty-dirs"]
    if dry_run:
        command.append("--dry-run")
    command.extend(["-e", f"ssh -p {source.port}"])
    for pattern in _default_include_patterns(include_videos, include_checkpoints) + list(source.include_patterns):
        command.append(f"--include={pattern}")
    for pattern in _default_exclude_patterns(include_videos, include_checkpoints) + list(source.exclude_patterns):
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


def _default_include_patterns(include_videos: bool, include_checkpoints: bool) -> List[str]:
    patterns = [
        "*/",
        "params/***",
        "events.out.tfevents*",
        "config.*",
    ]
    if include_checkpoints:
        patterns.append("model_*.pt")
    if include_videos:
        patterns.append("videos/***")
    return patterns


def _default_exclude_patterns(include_videos: bool, include_checkpoints: bool) -> List[str]:
    patterns = []
    if not include_checkpoints:
        patterns.append("model_*.pt")
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
