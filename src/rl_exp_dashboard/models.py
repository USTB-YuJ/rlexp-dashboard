from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ConfigDiff:
    path: str
    kind: str
    before: Any = None
    after: Any = None


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    iteration: Optional[int]
    size_bytes: int
    modified_time: float
    is_latest: bool = False


@dataclass(frozen=True)
class MetricSummary:
    tag: str
    first_step: int
    last_step: int
    first_value: float
    last_value: float
    min_value: float
    max_value: float
    count: int
    window_means: Dict[str, float] = field(default_factory=dict)
    slope_last_points: float = 0.0


@dataclass(frozen=True)
class MetricSeries:
    tag: str
    points: List[Dict[str, float]] = field(default_factory=list)
    original_count: int = 0


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    group: str
    path: Path
    modified_time: float
    start_time: float = 0.0
    task_name: str = ""
    algorithm_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    git_metadata: Dict[str, Any] = field(default_factory=dict)
    param_files: List[Path] = field(default_factory=list)
    event_files: List[Path] = field(default_factory=list)
    checkpoints: List[CheckpointRecord] = field(default_factory=list)
    metric_summaries: List[MetricSummary] = field(default_factory=list)
    metric_series: List[MetricSeries] = field(default_factory=list)
    videos: List[Path] = field(default_factory=list)
    artifacts: List[Path] = field(default_factory=list)
    parent_run_id: Optional[str] = None
    parent_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class LineageEdge:
    parent_run_id: str
    child_run_id: str
    relationship: str = "manual-link"
    parent_checkpoint: Optional[str] = None
    intended_change: str = ""
    note: str = ""
    confirmed: bool = True
    confirmation_state: Optional[str] = None
    confidence_source: str = "manual"
    result_summary: str = ""


def inferred_resume_lineage_edge(run: RunRecord) -> LineageEdge:
    if not run.parent_run_id:
        raise ValueError(f"Run `{run.run_id}` does not contain parent resume metadata.")
    return LineageEdge(
        parent_run_id=run.parent_run_id,
        child_run_id=run.run_id,
        relationship="resume",
        parent_checkpoint=run.parent_checkpoint,
        intended_change="Inferred from indexed resume/load_run parameters.",
        confirmation_state="confirmed",
        confidence_source="params",
    )
