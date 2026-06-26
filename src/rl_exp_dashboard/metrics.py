from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import MetricSummary


ScalarPoint = Tuple[int, float]


def summarize_scalar_series(
    tag: str,
    points: Iterable[ScalarPoint],
    windows: Sequence[int] = (100, 500, 1000, 5000),
    slope_points: int = 50,
) -> Optional[MetricSummary]:
    ordered = sorted((int(step), float(value)) for step, value in points)
    if not ordered:
        return None

    values = [value for _, value in ordered]
    last_step = ordered[-1][0]
    window_means = {}
    for window in windows:
        recent = [value for step, value in ordered if step >= last_step - window]
        if recent:
            window_means[f"mean{window}"] = sum(recent) / len(recent)

    return MetricSummary(
        tag=tag,
        first_step=ordered[0][0],
        last_step=last_step,
        first_value=ordered[0][1],
        last_value=ordered[-1][1],
        min_value=min(values),
        max_value=max(values),
        count=len(ordered),
        window_means=window_means,
        slope_last_points=_linear_slope(ordered[-slope_points:]),
    )


def _linear_slope(points: List[ScalarPoint]) -> float:
    if len(points) < 2:
        return 0.0

    xs = [step for step, _ in points]
    ys = [value for _, value in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return numerator / denominator


def read_tensorboard_metric_summaries(event_files: Sequence[Path]) -> List[MetricSummary]:
    """Read scalar summaries from TensorBoard event files when tensorboard is installed."""

    if not event_files:
        return []

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError:
        return []

    by_tag: Dict[str, List[ScalarPoint]] = {}
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={"scalars": 0},
        )
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            by_tag.setdefault(tag, []).extend((event.step, event.value) for event in accumulator.Scalars(tag))

    summaries = []
    for tag, points in sorted(by_tag.items()):
        summary = summarize_scalar_series(tag, points)
        if summary is not None:
            summaries.append(summary)
    return summaries
