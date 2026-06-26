from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def render_run_report_markdown(payload: Dict[str, Any]) -> str:
    run = payload.get("run")
    if not run:
        return "# Experiment Report\n\nRun not found.\n"

    lines: list[str] = [f"# Experiment Report: {run['run_id']}", ""]
    lines.extend(_run_summary(run))
    lines.extend(_lineage_summary(payload))
    lines.extend(_observation_summary(payload.get("observation") or {}))
    lines.extend(_metric_summary(payload.get("metrics") or []))
    lines.extend(_checkpoint_reviews(payload.get("checkpoint_reviews") or []))
    lines.extend(_artifact_summary(payload))
    return "\n".join(lines).rstrip() + "\n"


def _run_summary(run: Dict[str, Any]) -> list[str]:
    latest_checkpoint = run.get("latest_checkpoint") or ""
    return [
        "## Run",
        "",
        f"- Project: {_text(run.get('project_name'))}",
        f"- Group: {_text(run.get('group'))}",
        f"- Task: {_text(run.get('task_name'))}",
        f"- Algorithm: {_text(run.get('algorithm_name'))}",
        f"- Latest checkpoint: {_text(latest_checkpoint)}",
        f"- Path: {_text(run.get('path'))}",
        "",
    ]


def _lineage_summary(payload: Dict[str, Any]) -> list[str]:
    parent_compare = payload.get("parent_compare") or {}
    parent_lineage = payload.get("parent_lineage") or []
    parent_edge = parent_lineage[0] if parent_lineage else {}
    parent_run_id = parent_compare.get("parent_run_id") or parent_edge.get("parent_run_id") or ""
    parent_checkpoint = parent_compare.get("parent_checkpoint") or parent_edge.get("parent_checkpoint") or ""
    relationship = parent_compare.get("relationship") or parent_edge.get("relationship") or ""
    intended_change = parent_compare.get("intended_change") or parent_edge.get("intended_change") or ""
    result_summary = parent_edge.get("result_summary") or ""

    lines = [
        "## Lineage",
        "",
        f"- Parent: {_text(parent_run_id)}",
        f"- Parent checkpoint: {_text(parent_checkpoint)}",
        f"- Relationship: {_text(relationship)}",
        f"- Intended change: {_text(intended_change)}",
    ]
    if result_summary:
        lines.append(f"- Result summary: {_text(result_summary)}")
    lines.append("")
    return lines


def _observation_summary(observation: Dict[str, Any]) -> list[str]:
    tags = ", ".join(sorted(str(tag) for tag in observation.get("tags", []) if str(tag)))
    return [
        "## Manual Observation",
        "",
        f"- Verdict: {_text(observation.get('verdict') or 'unreviewed')}",
        f"- Summary: {_text(observation.get('summary'))}",
        f"- Tags: {_text(tags)}",
        f"- Recommended checkpoint: {_text(observation.get('recommended_checkpoint'))}",
        "",
    ]


def _metric_summary(metrics: list[Dict[str, Any]]) -> list[str]:
    lines = ["## Metrics", ""]
    if not metrics:
        return [*lines, "No metric summaries indexed.", ""]
    lines.extend(["| Tag | Last | Step | Mean1000 |", "| --- | ---: | ---: | ---: |"])
    for metric in metrics:
        window_means = metric.get("window_means") or {}
        lines.append(
            "| {tag} | {last} | {step} | {mean1000} |".format(
                tag=_cell(metric.get("tag")),
                last=_cell(metric.get("last_value")),
                step=_cell(metric.get("last_step")),
                mean1000=_cell(window_means.get("mean1000")),
            )
        )
    lines.append("")
    return lines


def _checkpoint_reviews(reviews: list[Dict[str, Any]]) -> list[str]:
    lines = ["## Checkpoint Reviews", ""]
    if not reviews:
        return [*lines, "No checkpoint reviews recorded.", ""]
    lines.extend(
        [
            "| Checkpoint | Status | Score | Recommended | Tags | Notes |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for review in reviews:
        lines.append(
            "| {checkpoint} | {status} | {score} | {recommended} | {tags} | {notes} |".format(
                checkpoint=_cell(review.get("checkpoint")),
                status=_cell(review.get("status")),
                score=_cell(review.get("score")),
                recommended="yes" if review.get("recommended") else "",
                tags=_cell(", ".join(review.get("tags") or [])),
                notes=_cell(review.get("notes")),
            )
        )
    lines.append("")
    return lines


def _artifact_summary(payload: Dict[str, Any]) -> list[str]:
    checkpoints = payload.get("checkpoints") or []
    artifacts = payload.get("artifacts") or []
    config_files = payload.get("config_files") or []
    event_files = payload.get("event_files") or []
    latest = next((item for item in checkpoints if item.get("is_latest")), None)
    return [
        "## Indexed Sources",
        "",
        f"- Checkpoints: {len(checkpoints)}",
        f"- Latest checkpoint file: {_text(Path(latest['path']).name if latest else '')}",
        f"- Videos: {sum(1 for item in artifacts if item.get('kind') == 'video')}",
        f"- Artifacts: {sum(1 for item in artifacts if item.get('kind') != 'video')}",
        f"- Config files: {len(config_files)}",
        f"- TensorBoard event files: {len(event_files)}",
        "",
    ]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _cell(value: Any) -> str:
    text = _text(value)
    return text.replace("|", "\\|").replace("\n", " ")
