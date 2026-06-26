from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional

from .api import serve
from .indexer import LocalRunIndexer
from .models import MetricSummary, inferred_resume_lineage_edge
from .project_config import ProjectConfig, load_project_config
from .storage import DashboardStore
from .sync import RemoteSource, SyncRunner, build_sync_plan, execute_sync_plan


def main(
    argv: Optional[List[str]] = None,
    metric_reader: Optional[Callable[[List[Path]], List[MetricSummary]]] = None,
    sync_runner: Optional[SyncRunner] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return _index(args, metric_reader=metric_reader)
    if args.command == "serve":
        serve(workspace=Path(args.workspace), host=args.host, port=args.port)
        return 0
    if args.command == "sync":
        return _sync(args, runner=sync_runner)
    if args.command == "project":
        if args.project_command == "import":
            return _import_project(args)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-exp-dashboard")
    subparsers = parser.add_subparsers(dest="command")

    index = subparsers.add_parser("index", help="Index a local RL log root into SQLite.")
    index.add_argument("--project", required=True, help="Project name to associate with discovered runs.")
    index.add_argument("--log-root", type=Path, help="Local log root to scan. Defaults to imported project cache root.")
    index.add_argument("--db", required=True, type=Path, help="SQLite database path.")

    serve_parser = subparsers.add_parser("serve", help="Start the local dashboard server.")
    serve_parser.add_argument("--workspace", default="~/rl-exp-dashboard", help="Workspace directory.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", default=7860, type=int, help="Bind port.")

    project_parser = subparsers.add_parser("project", help="Manage dashboard project configuration.")
    project_subparsers = project_parser.add_subparsers(dest="project_command")
    project_import = project_subparsers.add_parser("import", help="Import a project config file.")
    project_import.add_argument("--config", required=True, type=Path, help="Project config file.")
    project_import.add_argument("--db", required=True, type=Path, help="SQLite database path.")

    sync_parser = subparsers.add_parser("sync", help="Preview or run a remote log sync.")
    sync_parser.add_argument("--project", required=True, help="Project name.")
    sync_parser.add_argument("--source-name", required=True, help="Remote source display name.")
    sync_parser.add_argument("--host", help="SSH host. Defaults to imported remote source host.")
    sync_parser.add_argument("--user", help="SSH user. Defaults to imported remote source user.")
    sync_parser.add_argument("--port", type=int, help="SSH port. Defaults to imported remote source port or 22.")
    sync_parser.add_argument("--remote-log-root", help="Remote log root. Defaults to imported remote source root.")
    sync_parser.add_argument("--cache-root", type=Path, help="Local cache root. Defaults to imported project cache root.")
    sync_parser.add_argument("--db", required=True, type=Path, help="SQLite database path.")
    sync_parser.add_argument("--method", choices=["rsync", "scp", "ssh-tar"], help="Sync method.")
    sync_parser.add_argument("--include-videos", action="store_true", help="Include play videos in sync plan.")
    sync_parser.add_argument("--dry-run", action="store_true", help="Preview command and do not execute.")

    return parser


def _index(
    args: argparse.Namespace,
    metric_reader: Optional[Callable[[List[Path]], List[MetricSummary]]] = None,
) -> int:
    store = DashboardStore(args.db)
    store.initialize()
    log_root = _resolve_log_root(store, args.project, args.log_root)
    store.upsert_project(args.project, local_cache_root=log_root)

    runs = LocalRunIndexer(log_root, metric_reader=metric_reader).discover_runs()
    for run in runs:
        store.upsert_run(args.project, run)
        if run.parent_run_id:
            store.upsert_lineage(inferred_resume_lineage_edge(run))

    print(f"Indexed {len(runs)} runs into {args.db}")
    return 0


def _import_project(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    store = DashboardStore(args.db)
    store.initialize()
    _persist_project_config(store, config)
    print(f"Imported project {config.name} into {args.db}")
    return 0


def _persist_project_config(store: DashboardStore, config: ProjectConfig) -> None:
    store.upsert_project(
        config.name,
        config.local_cache_root,
        parser_profile=config.parser_profile,
        preferred_metrics=list(config.preferred_metrics),
        log_patterns=list(config.log_patterns),
        tag_schema=list(config.tag_schema),
    )
    for source in config.remote_sources:
        store.upsert_remote_source(source)


def _sync(args: argparse.Namespace, runner: Optional[SyncRunner] = None) -> int:
    store = DashboardStore(args.db)
    store.initialize()
    cache_root = _resolve_cache_root(store, args.project, args.cache_root)
    source = _resolve_remote_source(store, args)
    store.upsert_project(args.project, local_cache_root=cache_root)
    store.upsert_remote_source(source)
    plan = build_sync_plan(
        source,
        cache_root=cache_root,
        dry_run=args.dry_run,
        include_videos=args.include_videos,
    )
    print(" ".join(plan.command))
    if args.dry_run:
        store.record_sync_status(
            source_name=source.name,
            project=source.project,
            status="dry-run",
            command=plan.command,
            local_path=plan.local_path,
            message="preview only",
        )
        return 0

    result = execute_sync_plan(plan, runner=runner)
    message = result.stdout.strip() or result.stderr.strip()
    store.record_sync_status(
        source_name=source.name,
        project=source.project,
        status=result.status,
        command=result.command,
        local_path=plan.local_path,
        message=message,
    )
    print(f"Sync {result.status}: {message}".rstrip())
    return 0 if result.status == "completed" else result.return_code


def _resolve_log_root(store: DashboardStore, project_name: str, explicit_log_root: Optional[Path]) -> Path:
    if explicit_log_root is not None:
        return explicit_log_root
    project = store.get_project(project_name)
    if project is None:
        raise SystemExit(f"Project `{project_name}` has not been imported and --log-root was not provided.")
    return Path(project["local_cache_root"])


def _resolve_cache_root(store: DashboardStore, project_name: str, explicit_cache_root: Optional[Path]) -> Path:
    if explicit_cache_root is not None:
        return explicit_cache_root
    project = store.get_project(project_name)
    if project is None:
        raise SystemExit(f"Project `{project_name}` has not been imported and --cache-root was not provided.")
    return Path(project["local_cache_root"])


def _resolve_remote_source(store: DashboardStore, args: argparse.Namespace) -> RemoteSource:
    imported_source = next(
        (source for source in store.list_remote_sources(args.project) if source["name"] == args.source_name),
        None,
    )
    host = args.host or (imported_source or {}).get("host")
    user = args.user or (imported_source or {}).get("user")
    remote_log_root = args.remote_log_root or (imported_source or {}).get("remote_log_root")
    if not host or not user or not remote_log_root:
        raise SystemExit(
            f"Remote source `{args.source_name}` for project `{args.project}` has not been imported "
            "and --host/--user/--remote-log-root were not fully provided."
        )
    return RemoteSource(
        name=args.source_name,
        host=str(host),
        user=str(user),
        port=int(args.port if args.port is not None else (imported_source or {}).get("port", 22)),
        remote_log_root=str(remote_log_root),
        project=args.project,
        method=str(args.method or (imported_source or {}).get("method", "rsync")),
        include_patterns=tuple((imported_source or {}).get("include_patterns", [])),
        exclude_patterns=tuple((imported_source or {}).get("exclude_patterns", [])),
    )


if __name__ == "__main__":
    raise SystemExit(main())
