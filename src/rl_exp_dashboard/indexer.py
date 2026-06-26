from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import CheckpointRecord, RunRecord
from .structured_loader import load_structured_file


_CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
_ARTIFACT_SUFFIXES = {".onnx", ".jit", ".pt2"}
_PARAM_SUFFIXES = {".yaml", ".yml", ".json"}


class LocalRunIndexer:
    def __init__(self, log_root: Path):
        self.log_root = Path(log_root)

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
        videos = self._find_files_by_suffix(run_dir, _VIDEO_SUFFIXES)
        artifacts = self._find_files_by_suffix(run_dir, _ARTIFACT_SUFFIXES)

        return RunRecord(
            run_id=run_id,
            name=name,
            group=group,
            path=run_dir,
            modified_time=run_dir.stat().st_mtime,
            params=params,
            param_files=param_files,
            event_files=event_files,
            checkpoints=checkpoints,
            videos=videos,
            artifacts=artifacts,
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
