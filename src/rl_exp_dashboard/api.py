from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config_diff import diff_configs
from .models import LineageEdge
from .storage import DashboardStore

_DEFAULT_RUN_TABLE_METRICS = (
    "Train/mean_reward",
    "Episode/reward",
    "Episode/return",
    "Metrics/mean_reward",
)

_CONFIG_DIFF_GROUPS = (
    ("rewards", "Reward Diffs"),
    ("algorithm", "Algorithm Diffs"),
    ("observation_network", "Observation / Network Diffs"),
    ("curriculum_termination", "Curriculum / Termination Diffs"),
)


def runs_payload(store: DashboardStore, project: str | None = None) -> Dict[str, Any]:
    projects_by_name = {item["name"]: item for item in store.list_projects()}
    return {
        "runs": [
            _run_table_row(store, run, projects_by_name.get(run["project_name"]))
            for run in store.list_runs(project)
        ]
    }


def timeline_payload(store: DashboardStore, project: str | None = None) -> Dict[str, Any]:
    projects_by_name = {item["name"]: item for item in store.list_projects()}
    runs = sorted(
        store.list_runs(project),
        key=lambda run: (float(run.get("modified_time") or 0.0), run["run_id"]),
    )
    return {
        "project": project,
        "entries": [
            _timeline_entry(store, run, projects_by_name.get(run["project_name"]))
            for run in runs
        ],
    }


def metric_summaries_payload(store: DashboardStore, run_id: str) -> Dict[str, Any]:
    return {"run_id": run_id, "metrics": store.list_metric_summaries(run_id)}


def metric_series_payload(store: DashboardStore, run_id: str, tag: str | None = None) -> Dict[str, Any]:
    return {"run_id": run_id, "series": store.list_metric_series(run_id, tag=tag)}


def run_detail_payload(store: DashboardStore, run_id: str) -> Dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        return {
            "run": None,
            "checkpoints": [],
            "metrics": [],
            "artifacts": [],
            "parent_lineage": [],
            "child_lineage": [],
            "parent_compare": None,
        }
    parent_lineage = store.list_lineage(run_id)
    child_lineage = store.list_child_lineage(run_id)
    return {
        "run": run,
        "checkpoints": store.list_checkpoints(run_id),
        "artifacts": store.list_run_artifacts(run_id),
        "metrics": store.list_metric_summaries(run_id),
        "parent_lineage": parent_lineage,
        "child_lineage": child_lineage,
        "parent_compare": _parent_compare_summary(store, run, parent_lineage),
        "observation": store.get_run_observation(run_id),
        "checkpoint_reviews": store.list_checkpoint_reviews(run_id),
    }


def artifact_file_path(store: DashboardStore, run_id: str, path: str) -> Path:
    requested = Path(path).expanduser().resolve(strict=False)
    for artifact in store.list_run_artifacts(run_id):
        candidate = Path(artifact["path"]).expanduser()
        if candidate.resolve(strict=False) == requested:
            if not candidate.is_file():
                raise FileNotFoundError(str(candidate))
            return candidate
    raise PermissionError(f"Artifact is not indexed for run: {run_id}")


def compare_runs_payload(store: DashboardStore, before_run_id: str, after_run_id: str) -> Dict[str, Any]:
    before = store.get_run(before_run_id)
    after = store.get_run(after_run_id)
    before_params = before["params"] if before is not None else {}
    after_params = after["params"] if after is not None else {}
    config_diffs = [
        {"path": diff.path, "kind": diff.kind, "before": diff.before, "after": diff.after}
        for diff in diff_configs(before_params, after_params)
    ]
    return {
        "before": before,
        "after": after,
        "config_diffs": config_diffs,
        "config_diff_groups": _config_diff_groups(config_diffs),
        "metric_deltas": _metric_deltas(
            store.list_metric_summaries(before_run_id),
            store.list_metric_summaries(after_run_id),
        ),
    }


def _config_diff_groups(config_diffs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped = {key: {"key": key, "title": title, "diffs": []} for key, title in _CONFIG_DIFF_GROUPS}
    for diff in config_diffs:
        group_key = _config_diff_group_key(str(diff.get("path", "")))
        if group_key is not None:
            grouped[group_key]["diffs"].append(diff)
    return [grouped[key] for key, _title in _CONFIG_DIFF_GROUPS]


def _config_diff_group_key(path: str) -> str | None:
    normalized = path.lower()
    if normalized.startswith(("rewards.", "reward.")) or ".rewards." in normalized:
        return "rewards"
    if normalized.startswith(("algorithm.", "agent.algorithm.")) or ".algorithm." in normalized:
        return "algorithm"
    if normalized.startswith(("observations.", "observation.", "actor.", "critic.", "policy.")):
        return "observation_network"
    if (
        normalized.startswith(("agent.policy.", "network.", "model."))
        or ".network." in normalized
        or ".model." in normalized
        or "depth_encoder" in normalized
    ):
        return "observation_network"
    if normalized.startswith(("curriculum.", "curriculums.", "termination.", "terminations.")):
        return "curriculum_termination"
    if ".curriculum." in normalized or ".termination." in normalized or ".terminations." in normalized:
        return "curriculum_termination"
    return None


def _parent_compare_summary(
    store: DashboardStore,
    run: Dict[str, Any],
    parent_lineage: list[Dict[str, Any]],
) -> Dict[str, Any] | None:
    parent_edge = parent_lineage[0] if parent_lineage else {}
    parent_run_id = parent_edge.get("parent_run_id") or run.get("parent_run_id")
    if not parent_run_id or store.get_run(parent_run_id) is None:
        return None

    comparison = compare_runs_payload(store, parent_run_id, run["run_id"])
    return {
        "parent_run_id": parent_run_id,
        "child_run_id": run["run_id"],
        "relationship": parent_edge.get("relationship") or "resume",
        "parent_checkpoint": parent_edge.get("parent_checkpoint") or run.get("parent_checkpoint"),
        "intended_change": parent_edge.get("intended_change", ""),
        "note": parent_edge.get("note", ""),
        "config_diff_groups": comparison["config_diff_groups"],
        "config_diffs": comparison["config_diffs"],
        "metric_deltas": comparison["metric_deltas"],
    }


def remote_sources_payload(store: DashboardStore, project: str | None = None) -> Dict[str, Any]:
    sources = []
    for source in store.list_remote_sources(project):
        source = dict(source)
        source["latest_sync"] = store.latest_sync_status(source["name"], source["project"])
        sources.append(source)
    return {"sources": sources}


def projects_payload(store: DashboardStore) -> Dict[str, Any]:
    return {"projects": store.list_projects()}


def lineage_overview_payload(store: DashboardStore, project: str | None = None) -> Dict[str, Any]:
    nodes = [
        {
            "run_id": run["run_id"],
            "project_name": run["project_name"],
            "name": run["name"],
            "group": run["group"],
            "latest_checkpoint": run["latest_checkpoint"],
            "modified_time": run["modified_time"],
        }
        for run in store.list_runs(project)
    ]
    return {
        "project": project,
        "nodes": nodes,
        "edges": store.list_lineage_edges(project),
    }


def save_lineage_edge_payload(store: DashboardStore, payload: Dict[str, Any]) -> Dict[str, Any]:
    parent_run_id = str(payload["parent_run_id"])
    child_run_id = str(payload["child_run_id"])
    edge = LineageEdge(
        parent_run_id=parent_run_id,
        child_run_id=child_run_id,
        relationship=str(payload.get("relationship") or "manual-link"),
        parent_checkpoint=payload.get("parent_checkpoint") or None,
        intended_change=str(payload.get("intended_change", "")),
        note=str(payload.get("note", "")),
        confirmed=_payload_bool(payload.get("confirmed", True)),
    )
    store.upsert_lineage(edge)
    saved_edge = next(
        item for item in store.list_lineage(child_run_id) if item["parent_run_id"] == parent_run_id
    )
    return {"edge": saved_edge}


def save_run_observation_payload(store: DashboardStore, payload: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(payload["run_id"])
    store.upsert_run_observation(
        run_id=run_id,
        verdict=str(payload.get("verdict", "unreviewed")),
        summary=str(payload.get("summary", "")),
        tags=list(payload.get("tags", [])),
        recommended_checkpoint=payload.get("recommended_checkpoint") or None,
    )
    return {"run_id": run_id, "observation": store.get_run_observation(run_id)}


def save_checkpoint_review_payload(store: DashboardStore, payload: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(payload["run_id"])
    checkpoint = str(payload["checkpoint"])
    store.upsert_checkpoint_review(
        run_id=run_id,
        checkpoint=checkpoint,
        status=str(payload.get("status", "unreviewed")),
        notes=str(payload.get("notes", "")),
        tags=list(payload.get("tags", [])),
        video_path=str(payload.get("video_path", "")),
        score=payload.get("score"),
        recommended=bool(payload.get("recommended", False)),
    )
    review = next(
        item for item in store.list_checkpoint_reviews(run_id) if item["checkpoint"] == checkpoint
    )
    return {"run_id": run_id, "review": review}


def _payload_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _run_table_row(store: DashboardStore, run: Dict[str, Any], project: Dict[str, Any] | None) -> Dict[str, Any]:
    run_id = run["run_id"]
    artifacts = store.list_run_artifacts(run_id)
    observation = store.get_run_observation(run_id) or {}
    checkpoint_reviews = store.list_checkpoint_reviews(run_id)
    parent_lineage = store.list_lineage(run_id)
    child_lineage = store.list_child_lineage(run_id)
    row = dict(run)
    row.update(
        {
            "review_verdict": observation.get("verdict", "unreviewed"),
            "recommended_checkpoint": observation.get("recommended_checkpoint"),
            "has_observation": bool(observation),
            "has_reviewed_checkpoint": bool(checkpoint_reviews),
            "video_count": sum(1 for artifact in artifacts if artifact["kind"] == "video"),
            "artifact_count": sum(1 for artifact in artifacts if artifact["kind"] != "video"),
            "parent_count": len(parent_lineage),
            "child_count": len(child_lineage),
            "has_parent": bool(run.get("parent_run_id") or parent_lineage),
            "has_children": bool(child_lineage),
            "key_metrics": _run_table_metrics(
                store.list_metric_summaries(run_id),
                project.get("preferred_metrics", []) if project else [],
            ),
        }
    )
    return row


def _timeline_entry(store: DashboardStore, run: Dict[str, Any], project: Dict[str, Any] | None) -> Dict[str, Any]:
    run_id = run["run_id"]
    observation = store.get_run_observation(run_id) or {}
    parent_lineage = store.list_lineage(run_id)
    parent_edge = parent_lineage[0] if parent_lineage else {}
    return {
        "run_id": run_id,
        "project_name": run["project_name"],
        "name": run["name"],
        "group": run["group"],
        "modified_time": run["modified_time"],
        "latest_checkpoint": run["latest_checkpoint"],
        "parent_run_id": parent_edge.get("parent_run_id") or run.get("parent_run_id"),
        "parent_checkpoint": parent_edge.get("parent_checkpoint") or run.get("parent_checkpoint"),
        "relationship": parent_edge.get("relationship", ""),
        "intended_change": parent_edge.get("intended_change", ""),
        "review_verdict": observation.get("verdict", "unreviewed"),
        "summary": observation.get("summary", ""),
        "tags": observation.get("tags", []),
        "recommended_checkpoint": observation.get("recommended_checkpoint"),
        "key_metrics": _run_table_metrics(
            store.list_metric_summaries(run_id),
            project.get("preferred_metrics", []) if project else [],
        ),
    }


def _run_table_metrics(metrics: list[Dict[str, Any]], preferred_tags: list[str]) -> Dict[str, Dict[str, Any]]:
    metrics_by_tag = {metric["tag"]: metric for metric in metrics}
    ordered_tags = list(preferred_tags) or list(_DEFAULT_RUN_TABLE_METRICS)
    selected = [metrics_by_tag[tag] for tag in ordered_tags if tag in metrics_by_tag]
    if not selected and metrics:
        selected = [metrics[0]]
    return {metric["tag"]: _compact_metric_summary(metric) for metric in selected}


def _compact_metric_summary(metric: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tag": metric["tag"],
        "last_value": metric.get("last_value"),
        "last_step": metric.get("last_step"),
        "window_means": metric.get("window_means", {}),
        "slope_last_points": metric.get("slope_last_points"),
    }


def _metric_deltas(before_metrics: list[Dict[str, Any]], after_metrics: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    before_by_tag = {metric["tag"]: metric for metric in before_metrics}
    after_by_tag = {metric["tag"]: metric for metric in after_metrics}
    deltas = []
    for tag in sorted(set(before_by_tag) | set(after_by_tag)):
        before_value = before_by_tag.get(tag, {}).get("last_value")
        after_value = after_by_tag.get(tag, {}).get("last_value")
        delta = None
        if before_value is not None and after_value is not None:
            delta = after_value - before_value
        deltas.append(
            {
                "tag": tag,
                "before_last_value": before_value,
                "after_last_value": after_value,
                "delta_last_value": delta,
            }
        )
    return deltas


def create_app(db_path: Path):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI is required to serve the dashboard. Install with "
            "`pip install rl-exp-dashboard[server]`."
        ) from exc

    app = FastAPI(title="RL Experiment Dashboard")
    store = DashboardStore(db_path)
    store.initialize()
    static_dir = Path(__file__).with_name("web_static")

    @app.get("/api/runs")
    def list_runs(project: str | None = None):
        return runs_payload(store, project)

    @app.get("/api/projects")
    def list_projects():
        return projects_payload(store)

    @app.get("/api/remote-sources")
    def list_remote_sources(project: str | None = None):
        return remote_sources_payload(store, project)

    @app.get("/api/lineage")
    def get_lineage(project: str | None = None):
        return lineage_overview_payload(store, project)

    @app.get("/api/timeline")
    def get_timeline(project: str | None = None):
        return timeline_payload(store, project)

    @app.post("/api/lineage-edge")
    async def save_lineage_edge(payload: dict):
        return save_lineage_edge_payload(store, payload)

    @app.post("/api/run-observation")
    async def save_run_observation(payload: dict):
        return save_run_observation_payload(store, payload)

    @app.post("/api/checkpoint-review")
    async def save_checkpoint_review(payload: dict):
        return save_checkpoint_review_payload(store, payload)

    @app.get("/api/run-detail")
    def get_run_detail(run_id: str):
        return run_detail_payload(store, run_id)

    @app.get("/api/metrics")
    def list_metrics(run_id: str):
        return metric_summaries_payload(store, run_id)

    @app.get("/api/metric-series")
    def list_metric_series(run_id: str, tag: str | None = None):
        return metric_series_payload(store, run_id, tag=tag)

    @app.get("/api/artifact-file")
    def get_artifact_file(run_id: str, path: str):
        try:
            return FileResponse(artifact_file_path(store, run_id, path))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id:path}")
    def get_run(run_id: str):
        return run_detail_payload(store, run_id)

    @app.get("/api/compare")
    def compare_runs(before_run_id: str, after_run_id: str):
        return compare_runs_payload(store, before_run_id, after_run_id)

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def serve(workspace: Path, host: str = "127.0.0.1", port: int = 7860) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "uvicorn is required to serve the dashboard. Install with "
            "`pip install rl-exp-dashboard[server]`."
        ) from exc

    workspace = Path(workspace).expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    app = create_app(workspace / "dashboard.sqlite3")
    uvicorn.run(app, host=host, port=port)
