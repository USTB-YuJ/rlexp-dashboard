from __future__ import annotations

from pathlib import Path

from .storage import DashboardStore


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
