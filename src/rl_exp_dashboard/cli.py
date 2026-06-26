from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional

from .api import serve
from .indexer import LocalRunIndexer
from .models import MetricSummary
from .storage import DashboardStore
from .sync import RemoteSource, build_sync_plan


def main(
    argv: Optional[List[str]] = None,
    metric_reader: Optional[Callable[[List[Path]], List[MetricSummary]]] = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return _index(args, metric_reader=metric_reader)
    if args.command == "serve":
        serve(workspace=Path(args.workspace), host=args.host, port=args.port)
        return 0
    if args.command == "sync":
        return _sync(args)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-exp-dashboard")
    subparsers = parser.add_subparsers(dest="command")

    index = subparsers.add_parser("index", help="Index a local RL log root into SQLite.")
    index.add_argument("--project", required=True, help="Project name to associate with discovered runs.")
    index.add_argument("--log-root", required=True, type=Path, help="Local log root to scan.")
    index.add_argument("--db", required=True, type=Path, help="SQLite database path.")

    serve_parser = subparsers.add_parser("serve", help="Start the local dashboard server.")
    serve_parser.add_argument("--workspace", default="~/rl-exp-dashboard", help="Workspace directory.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", default=7860, type=int, help="Bind port.")

    sync_parser = subparsers.add_parser("sync", help="Preview or run a remote log sync.")
    sync_parser.add_argument("--project", required=True, help="Project name.")
    sync_parser.add_argument("--source-name", required=True, help="Remote source display name.")
    sync_parser.add_argument("--host", required=True, help="SSH host.")
    sync_parser.add_argument("--user", required=True, help="SSH user.")
    sync_parser.add_argument("--port", default=22, type=int, help="SSH port.")
    sync_parser.add_argument("--remote-log-root", required=True, help="Remote log root.")
    sync_parser.add_argument("--cache-root", required=True, type=Path, help="Local cache root.")
    sync_parser.add_argument("--db", required=True, type=Path, help="SQLite database path.")
    sync_parser.add_argument("--method", default="rsync", choices=["rsync", "scp", "ssh-tar"], help="Sync method.")
    sync_parser.add_argument("--include-videos", action="store_true", help="Include play videos in sync plan.")
    sync_parser.add_argument("--dry-run", action="store_true", help="Preview command and do not execute.")

    return parser


def _index(
    args: argparse.Namespace,
    metric_reader: Optional[Callable[[List[Path]], List[MetricSummary]]] = None,
) -> int:
    store = DashboardStore(args.db)
    store.initialize()
    store.upsert_project(args.project, local_cache_root=args.log_root)

    runs = LocalRunIndexer(args.log_root, metric_reader=metric_reader).discover_runs()
    for run in runs:
        store.upsert_run(args.project, run)

    print(f"Indexed {len(runs)} runs into {args.db}")
    return 0


def _sync(args: argparse.Namespace) -> int:
    store = DashboardStore(args.db)
    store.initialize()
    store.upsert_project(args.project, local_cache_root=args.cache_root)
    source = RemoteSource(
        name=args.source_name,
        host=args.host,
        user=args.user,
        port=args.port,
        remote_log_root=args.remote_log_root,
        project=args.project,
        method=args.method,
    )
    store.upsert_remote_source(source)
    plan = build_sync_plan(
        source,
        cache_root=args.cache_root,
        dry_run=args.dry_run,
        include_videos=args.include_videos,
    )
    status = "dry-run" if args.dry_run else "planned"
    store.record_sync_status(
        source_name=source.name,
        project=source.project,
        status=status,
        command=plan.command,
        local_path=plan.local_path,
        message="preview only" if args.dry_run else "execution not implemented",
    )
    print(" ".join(plan.command))
    if not args.dry_run:
        print("Sync execution is not implemented yet; rerun with --dry-run for preview.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
