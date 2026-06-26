from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .api import serve
from .indexer import LocalRunIndexer
from .storage import DashboardStore


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return _index(args)
    if args.command == "serve":
        serve(workspace=Path(args.workspace), host=args.host, port=args.port)
        return 0

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

    return parser


def _index(args: argparse.Namespace) -> int:
    store = DashboardStore(args.db)
    store.initialize()
    store.upsert_project(args.project, local_cache_root=args.log_root)

    runs = LocalRunIndexer(args.log_root).discover_runs()
    for run in runs:
        store.upsert_run(args.project, run)

    print(f"Indexed {len(runs)} runs into {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
