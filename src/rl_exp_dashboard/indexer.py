from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .metrics import read_tensorboard_metric_series, read_tensorboard_metric_summaries
from .models import CheckpointRecord, MetricSeries, MetricSummary, RunRecord
from .structured_loader import load_structured_file


_CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
_RUN_START_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[_-](?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})"
)
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
_ARTIFACT_SUFFIXES = {".onnx", ".jit", ".pt2"}
_PARAM_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
_PARENT_RUN_KEYS = {"load_run", "resume_run", "parent_run", "parent_run_id"}
_PARENT_CHECKPOINT_KEYS = {"load_checkpoint", "resume_checkpoint", "parent_checkpoint"}
_GIT_CONTAINER_KEYS = {"git", "git_info", "git_metadata"}
_TASK_NAME_KEYS = {"task", "task_name", "task_id", "env_name"}
_ALGORITHM_NAME_KEYS = {"algorithm", "algorithm_name", "algo", "algo_name"}
_NAME_KEYS = {"name", "id", "class_name"}


class LocalRunIndexer:
    def __init__(
        self,
        log_root: Path,
        metric_reader: Callable[[List[Path]], List[MetricSummary]] | None = None,
        metric_series_reader: Callable[[List[Path]], List[MetricSeries]] | None = None,
    ):
        self.log_root = Path(log_root)
        self.metric_reader = metric_reader or read_tensorboard_metric_summaries
        self.metric_series_reader = metric_series_reader or read_tensorboard_metric_series

    def discover_runs(self) -> List[RunRecord]:
        if not self.log_root.exists():
            return []

        runs = []
        for directory in sorted(path for path in self.log_root.rglob("*") if path.is_dir()):
            if self._looks_like_run_dir(directory):
                runs.append(self._index_run(directory))
        return runs

    def _looks_like_run_dir(self, directory: Path) -> bool:
        if (directory / "params").is_dir():
            return True
        if any(directory.glob("events.out.tfevents*")):
            return True
        if any(directory.glob("model_*.pt")):
            return True
        return False

    def _index_run(self, run_dir: Path) -> RunRecord:
        group = run_dir.parent.name
        name = run_dir.name
        run_id = f"{group}/{name}"
        param_files = self._find_param_files(run_dir)
        params = self._load_params(param_files)
        checkpoints = self._find_checkpoints(run_dir)
        event_files = sorted(run_dir.glob("events.out.tfevents*"))
        metric_summaries = self.metric_reader(event_files) if event_files else []
        metric_series = self.metric_series_reader(event_files) if event_files else []
        videos = self._find_files_by_suffix(run_dir, _VIDEO_SUFFIXES)
        artifacts = self._find_files_by_suffix(run_dir, _ARTIFACT_SUFFIXES)
        parent_run_id, parent_checkpoint = _infer_parent(params, group)
        git_metadata = _extract_git_metadata(params)
        task_name = _infer_named_metadata(params, _TASK_NAME_KEYS)
        algorithm_name = _infer_named_metadata(params, _ALGORITHM_NAME_KEYS)

        return RunRecord(
            run_id=run_id,
            name=name,
            group=group,
            path=run_dir,
            modified_time=run_dir.stat().st_mtime,
            start_time=_infer_start_time(run_dir),
            task_name=task_name,
            algorithm_name=algorithm_name,
            params=params,
            git_metadata=git_metadata,
            param_files=param_files,
            event_files=event_files,
            checkpoints=checkpoints,
            metric_summaries=metric_summaries,
            metric_series=metric_series,
            videos=videos,
            artifacts=artifacts,
            parent_run_id=parent_run_id,
            parent_checkpoint=parent_checkpoint,
        )

    def _find_param_files(self, run_dir: Path) -> List[Path]:
        params_dir = run_dir / "params"
        candidates: List[Path] = []
        if params_dir.is_dir():
            candidates.extend(
                path for path in params_dir.rglob("*") if path.is_file() and path.suffix.lower() in _PARAM_SUFFIXES
            )
        candidates.extend(
            path
            for path in run_dir.glob("config.*")
            if path.is_file() and path.suffix.lower() in _PARAM_SUFFIXES
        )
        return sorted(candidates)

    def _load_params(self, param_files: Iterable[Path]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for path in param_files:
            stem = path.stem
            loaded = load_structured_file(path)
            if set(loaded) == {stem} and isinstance(loaded[stem], dict):
                params[stem] = loaded[stem]
            else:
                params[stem] = loaded
        return params

    def _find_checkpoints(self, run_dir: Path) -> List[CheckpointRecord]:
        records = []
        for path in sorted(run_dir.glob("model_*.pt"), key=_checkpoint_sort_key):
            match = _CHECKPOINT_RE.match(path.name)
            iteration = int(match.group(1)) if match else None
            stat = path.stat()
            records.append(
                CheckpointRecord(
                    path=path,
                    iteration=iteration,
                    size_bytes=stat.st_size,
                    modified_time=stat.st_mtime,
                    is_latest=False,
                )
            )

        if records:
            latest = records[-1]
            records[-1] = CheckpointRecord(
                path=latest.path,
                iteration=latest.iteration,
                size_bytes=latest.size_bytes,
                modified_time=latest.modified_time,
                is_latest=True,
            )
        return records

    def _find_files_by_suffix(self, run_dir: Path, suffixes: set[str]) -> List[Path]:
        return sorted(path for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _checkpoint_sort_key(path: Path) -> int:
    match = _CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def _infer_start_time(run_dir: Path) -> float:
    match = _RUN_START_RE.search(run_dir.name)
    if not match:
        return 0.0
    timestamp = (
        f"{match.group('date')} "
        f"{match.group('hour')}:{match.group('minute')}:{match.group('second')}"
    )
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
    except ValueError:
        return 0.0


def _infer_parent(params: Dict[str, Any], group: str) -> tuple[str | None, str | None]:
    parent_run = _find_nested_value(params, _PARENT_RUN_KEYS)
    parent_checkpoint = _find_nested_value(params, _PARENT_CHECKPOINT_KEYS)
    if not parent_run:
        return None, _normalize_optional_string(parent_checkpoint)

    parent_run_text = str(parent_run).strip()
    if not parent_run_text or parent_run_text.lower() in {"none", "null", "false"}:
        return None, _normalize_optional_string(parent_checkpoint)
    if "/" not in parent_run_text:
        parent_run_text = f"{group}/{parent_run_text}"
    return parent_run_text, _normalize_optional_string(parent_checkpoint)


def _find_nested_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in keys:
                normalized = _normalize_optional_string(child)
                if normalized is not None:
                    return normalized
        for child in value.values():
            found = _find_nested_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_nested_value(child, keys)
            if found is not None:
                return found
    return None


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _extract_git_metadata(params: Dict[str, Any]) -> Dict[str, Any]:
    git = _find_git_mapping(params)
    if not git:
        return {}
    metadata: Dict[str, Any] = {}
    commit = git.get("commit") or git.get("sha") or git.get("hash")
    if commit is not None:
        metadata["commit"] = str(commit)
    branch = git.get("branch")
    if branch is not None:
        metadata["branch"] = str(branch)
    if "dirty" in git:
        metadata["dirty"] = git["dirty"]
    diff = git.get("diff") or git.get("dirty_diff")
    if diff is not None:
        metadata["diff"] = str(diff)
    return metadata


def _infer_named_metadata(params: Dict[str, Any], keys: set[str]) -> str:
    value = _find_named_value(params, keys)
    return str(value).strip() if value is not None else ""


def _find_named_value(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key in keys:
                if isinstance(child, dict):
                    named_child = _find_nested_value(child, _NAME_KEYS)
                    if named_child is not None:
                        return named_child
                else:
                    normalized = _normalize_optional_string(child)
                    if normalized is not None:
                        return normalized
        for child in value.values():
            found = _find_named_value(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_named_value(child, keys)
            if found is not None:
                return found
    return None


def _find_git_mapping(value: Any) -> Dict[str, Any] | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _GIT_CONTAINER_KEYS and isinstance(child, dict):
                return {str(child_key): child_value for child_key, child_value in child.items()}
        for child in value.values():
            found = _find_git_mapping(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_git_mapping(child)
            if found is not None:
                return found
    return None
