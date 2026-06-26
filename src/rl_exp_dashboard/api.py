from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config_diff import diff_configs
from .storage import DashboardStore


def metric_summaries_payload(store: DashboardStore, run_id: str) -> Dict[str, Any]:
    return {"run_id": run_id, "metrics": store.list_metric_summaries(run_id)}


def metric_series_payload(store: DashboardStore, run_id: str, tag: str | None = None) -> Dict[str, Any]:
    return {"run_id": run_id, "series": store.list_metric_series(run_id, tag=tag)}


def run_detail_payload(store: DashboardStore, run_id: str) -> Dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        return {"run": None, "checkpoints": [], "metrics": [], "parent_lineage": [], "child_lineage": []}
    return {
        "run": run,
        "checkpoints": store.list_checkpoints(run_id),
        "metrics": store.list_metric_summaries(run_id),
        "parent_lineage": store.list_lineage(run_id),
        "child_lineage": store.list_child_lineage(run_id),
        "observation": store.get_run_observation(run_id),
        "checkpoint_reviews": store.list_checkpoint_reviews(run_id),
    }


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
        "metric_deltas": _metric_deltas(
            store.list_metric_summaries(before_run_id),
            store.list_metric_summaries(after_run_id),
        ),
    }


def remote_sources_payload(store: DashboardStore, project: str | None = None) -> Dict[str, Any]:
    sources = []
    for source in store.list_remote_sources(project):
        source = dict(source)
        source["latest_sync"] = store.latest_sync_status(source["name"], source["project"])
        sources.append(source)
    return {"sources": sources}


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
        from fastapi import FastAPI
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
        return {"runs": store.list_runs(project)}

    @app.get("/api/remote-sources")
    def list_remote_sources(project: str | None = None):
        return remote_sources_payload(store, project)

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
