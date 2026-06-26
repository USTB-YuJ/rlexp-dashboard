from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LineageEdge, RunRecord


class DashboardStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists projects (
                    name text primary key,
                    local_cache_root text not null
                );

                create table if not exists runs (
                    run_id text primary key,
                    project_name text not null references projects(name),
                    name text not null,
                    run_group text not null,
                    path text not null,
                    modified_time real not null,
                    params_json text not null,
                    latest_checkpoint text,
                    parent_run_id text,
                    parent_checkpoint text
                );

                create table if not exists checkpoints (
                    run_id text not null references runs(run_id),
                    path text not null,
                    iteration integer,
                    size_bytes integer not null,
                    modified_time real not null,
                    is_latest integer not null,
                    primary key (run_id, path)
                );

                create table if not exists lineage_edges (
                    parent_run_id text not null,
                    child_run_id text not null,
                    relationship text not null,
                    parent_checkpoint text,
                    intended_change text not null,
                    note text not null,
                    confirmed integer not null,
                    primary key (parent_run_id, child_run_id)
                );
                """
            )

    def upsert_project(self, name: str, local_cache_root: Path) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into projects (name, local_cache_root)
                values (?, ?)
                on conflict(name) do update set local_cache_root=excluded.local_cache_root
                """,
                (name, str(local_cache_root)),
            )

    def upsert_run(self, project_name: str, run: RunRecord) -> None:
        latest_checkpoint = next((checkpoint.path.name for checkpoint in run.checkpoints if checkpoint.is_latest), None)
        with self._connect() as conn:
            conn.execute(
                """
                insert into runs (
                    run_id, project_name, name, run_group, path, modified_time, params_json,
                    latest_checkpoint, parent_run_id, parent_checkpoint
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(run_id) do update set
                    project_name=excluded.project_name,
                    name=excluded.name,
                    run_group=excluded.run_group,
                    path=excluded.path,
                    modified_time=excluded.modified_time,
                    params_json=excluded.params_json,
                    latest_checkpoint=excluded.latest_checkpoint,
                    parent_run_id=excluded.parent_run_id,
                    parent_checkpoint=excluded.parent_checkpoint
                """,
                (
                    run.run_id,
                    project_name,
                    run.name,
                    run.group,
                    str(run.path),
                    run.modified_time,
                    json.dumps(run.params, sort_keys=True),
                    latest_checkpoint,
                    run.parent_run_id,
                    run.parent_checkpoint,
                ),
            )
            conn.execute("delete from checkpoints where run_id = ?", (run.run_id,))
            conn.executemany(
                """
                insert into checkpoints (run_id, path, iteration, size_bytes, modified_time, is_latest)
                values (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        str(checkpoint.path),
                        checkpoint.iteration,
                        checkpoint.size_bytes,
                        checkpoint.modified_time,
                        int(checkpoint.is_latest),
                    )
                    for checkpoint in run.checkpoints
                ],
            )

    def upsert_lineage(self, edge: LineageEdge) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into lineage_edges (
                    parent_run_id, child_run_id, relationship, parent_checkpoint,
                    intended_change, note, confirmed
                )
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(parent_run_id, child_run_id) do update set
                    relationship=excluded.relationship,
                    parent_checkpoint=excluded.parent_checkpoint,
                    intended_change=excluded.intended_change,
                    note=excluded.note,
                    confirmed=excluded.confirmed
                """,
                (
                    edge.parent_run_id,
                    edge.child_run_id,
                    edge.relationship,
                    edge.parent_checkpoint,
                    edge.intended_change,
                    edge.note,
                    int(edge.confirmed),
                ),
            )

    def list_runs(self, project_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "select * from runs"
        params: tuple[Any, ...] = ()
        if project_name is not None:
            query += " where project_name = ?"
            params = (project_name,)
        query += " order by modified_time desc"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._run_row_to_dict(row) for row in rows]

    def list_checkpoints(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from checkpoints where run_id = ? order by iteration",
                (run_id,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "path": row["path"],
                "iteration": row["iteration"],
                "size_bytes": row["size_bytes"],
                "modified_time": row["modified_time"],
                "is_latest": bool(row["is_latest"]),
            }
            for row in rows
        ]

    def list_lineage(self, child_run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from lineage_edges where child_run_id = ? order by parent_run_id",
                (child_run_id,),
            ).fetchall()
        return [
            {
                "parent_run_id": row["parent_run_id"],
                "child_run_id": row["child_run_id"],
                "relationship": row["relationship"],
                "parent_checkpoint": row["parent_checkpoint"],
                "intended_change": row["intended_change"],
                "note": row["note"],
                "confirmed": bool(row["confirmed"]),
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _run_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "project_name": row["project_name"],
            "name": row["name"],
            "group": row["run_group"],
            "path": row["path"],
            "modified_time": row["modified_time"],
            "params": json.loads(row["params_json"]),
            "latest_checkpoint": row["latest_checkpoint"],
            "parent_run_id": row["parent_run_id"],
            "parent_checkpoint": row["parent_checkpoint"],
        }
