from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LineageEdge, RunRecord
from .sync import RemoteSource


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
                    local_cache_root text not null,
                    parser_profile text not null default 'generic_tensorboard',
                    preferred_metrics_json text not null default '[]',
                    log_patterns_json text not null default '[]',
                    tag_schema_json text not null default '[]'
                );

                create table if not exists remote_sources (
                    name text not null,
                    project_name text not null references projects(name),
                    host text not null,
                    user text not null,
                    port integer not null,
                    remote_log_root text not null,
                    method text not null,
                    include_patterns_json text not null default '[]',
                    exclude_patterns_json text not null default '[]',
                    primary key (name, project_name)
                );

                create table if not exists sync_status (
                    id integer primary key autoincrement,
                    source_name text not null,
                    project_name text not null,
                    status text not null,
                    command_json text not null,
                    local_path text not null,
                    message text not null,
                    created_at real not null
                );

                create table if not exists runs (
                    run_id text primary key,
                    project_name text not null references projects(name),
                    name text not null,
                    run_group text not null,
                    path text not null,
                    modified_time real not null,
                    params_json text not null,
                    git_json text not null default '{}',
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

                create table if not exists run_artifacts (
                    run_id text not null references runs(run_id),
                    kind text not null,
                    path text not null,
                    name text not null,
                    suffix text not null,
                    size_bytes integer not null,
                    modified_time real not null,
                    primary key (run_id, path)
                );

                create table if not exists metric_summaries (
                    run_id text not null references runs(run_id),
                    tag text not null,
                    first_step integer not null,
                    last_step integer not null,
                    first_value real not null,
                    last_value real not null,
                    min_value real not null,
                    max_value real not null,
                    count integer not null,
                    window_means_json text not null,
                    slope_last_points real not null,
                    primary key (run_id, tag)
                );

                create table if not exists metric_series (
                    run_id text not null references runs(run_id),
                    tag text not null,
                    points_json text not null,
                    original_count integer not null,
                    sampled_count integer not null,
                    first_step integer,
                    last_step integer,
                    primary key (run_id, tag)
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

                create table if not exists run_observations (
                    run_id text primary key references runs(run_id),
                    verdict text not null,
                    summary text not null,
                    tags_json text not null,
                    recommended_checkpoint text,
                    updated_at real not null
                );

                create table if not exists checkpoint_reviews (
                    run_id text not null references runs(run_id),
                    checkpoint text not null,
                    status text not null,
                    notes text not null,
                    tags_json text not null,
                    video_path text not null,
                    score real,
                    recommended integer not null,
                    updated_at real not null,
                    primary key (run_id, checkpoint)
                );
                """
            )
            _ensure_column(conn, "projects", "parser_profile", "text not null default 'generic_tensorboard'")
            _ensure_column(conn, "projects", "preferred_metrics_json", "text not null default '[]'")
            _ensure_column(conn, "projects", "log_patterns_json", "text not null default '[]'")
            _ensure_column(conn, "projects", "tag_schema_json", "text not null default '[]'")
            _ensure_column(conn, "remote_sources", "include_patterns_json", "text not null default '[]'")
            _ensure_column(conn, "remote_sources", "exclude_patterns_json", "text not null default '[]'")
            _ensure_column(conn, "runs", "git_json", "text not null default '{}'")

    def upsert_project(
        self,
        name: str,
        local_cache_root: Path,
        parser_profile: Optional[str] = None,
        preferred_metrics: Optional[List[str]] = None,
        log_patterns: Optional[List[str]] = None,
        tag_schema: Optional[List[str]] = None,
    ) -> None:
        existing = self.get_project(name)
        parser_profile = parser_profile or (existing or {}).get("parser_profile", "generic_tensorboard")
        preferred_metrics = preferred_metrics if preferred_metrics is not None else (existing or {}).get("preferred_metrics", [])
        log_patterns = log_patterns if log_patterns is not None else (existing or {}).get("log_patterns", [])
        tag_schema = tag_schema if tag_schema is not None else (existing or {}).get("tag_schema", [])
        with self._connect() as conn:
            conn.execute(
                """
                insert into projects (
                    name, local_cache_root, parser_profile, preferred_metrics_json,
                    log_patterns_json, tag_schema_json
                )
                values (?, ?, ?, ?, ?, ?)
                on conflict(name) do update set
                    local_cache_root=excluded.local_cache_root,
                    parser_profile=excluded.parser_profile,
                    preferred_metrics_json=excluded.preferred_metrics_json,
                    log_patterns_json=excluded.log_patterns_json,
                    tag_schema_json=excluded.tag_schema_json
                """,
                (
                    name,
                    str(local_cache_root),
                    parser_profile,
                    json.dumps(preferred_metrics or []),
                    json.dumps(log_patterns or []),
                    json.dumps(tag_schema or []),
                ),
            )

    def list_projects(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("select * from projects order by name").fetchall()
        return [self._project_row_to_dict(row) for row in rows]

    def get_project(self, name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("select * from projects where name = ?", (name,)).fetchone()
        if row is None:
            return None
        return self._project_row_to_dict(row)

    def upsert_remote_source(self, source: RemoteSource) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into remote_sources (
                    name, project_name, host, user, port, remote_log_root, method,
                    include_patterns_json, exclude_patterns_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(name, project_name) do update set
                    host=excluded.host,
                    user=excluded.user,
                    port=excluded.port,
                    remote_log_root=excluded.remote_log_root,
                    method=excluded.method,
                    include_patterns_json=excluded.include_patterns_json,
                    exclude_patterns_json=excluded.exclude_patterns_json
                """,
                (
                    source.name,
                    source.project,
                    source.host,
                    source.user,
                    source.port,
                    source.remote_log_root,
                    source.method,
                    json.dumps(list(source.include_patterns)),
                    json.dumps(list(source.exclude_patterns)),
                ),
            )

    def list_remote_sources(self, project_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "select * from remote_sources"
        params: tuple[Any, ...] = ()
        if project_name is not None:
            query += " where project_name = ?"
            params = (project_name,)
        query += " order by name"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "name": row["name"],
                "project": row["project_name"],
                "host": row["host"],
                "user": row["user"],
                "port": row["port"],
                "remote_log_root": row["remote_log_root"],
                "method": row["method"],
                "include_patterns": json.loads(row["include_patterns_json"]),
                "exclude_patterns": json.loads(row["exclude_patterns_json"]),
            }
            for row in rows
        ]

    def record_sync_status(
        self,
        source_name: str,
        project: str,
        status: str,
        command: List[str],
        local_path: Path,
        message: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into sync_status (
                    source_name, project_name, status, command_json, local_path, message, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_name,
                    project,
                    status,
                    json.dumps(command),
                    str(local_path),
                    message,
                    time.time(),
                ),
            )

    def latest_sync_status(self, source_name: str, project: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                select * from sync_status
                where source_name = ? and project_name = ?
                order by created_at desc, id desc
                limit 1
                """,
                (source_name, project),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_name": row["source_name"],
            "project": row["project_name"],
            "status": row["status"],
            "command": json.loads(row["command_json"]),
            "local_path": row["local_path"],
            "message": row["message"],
            "created_at": row["created_at"],
        }

    def upsert_run(self, project_name: str, run: RunRecord) -> None:
        latest_checkpoint = next((checkpoint.path.name for checkpoint in run.checkpoints if checkpoint.is_latest), None)
        with self._connect() as conn:
            conn.execute(
                """
                insert into runs (
                    run_id, project_name, name, run_group, path, modified_time, params_json,
                    git_json, latest_checkpoint, parent_run_id, parent_checkpoint
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(run_id) do update set
                    project_name=excluded.project_name,
                    name=excluded.name,
                    run_group=excluded.run_group,
                    path=excluded.path,
                    modified_time=excluded.modified_time,
                    params_json=excluded.params_json,
                    git_json=excluded.git_json,
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
                    json.dumps(run.git_metadata, sort_keys=True),
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
            conn.execute("delete from run_artifacts where run_id = ?", (run.run_id,))
            conn.executemany(
                """
                insert into run_artifacts (
                    run_id, kind, path, name, suffix, size_bytes, modified_time
                )
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        artifact["kind"],
                        artifact["path"],
                        artifact["name"],
                        artifact["suffix"],
                        artifact["size_bytes"],
                        artifact["modified_time"],
                    )
                    for artifact in _run_artifacts(run)
                ],
            )
            conn.execute("delete from metric_summaries where run_id = ?", (run.run_id,))
            conn.executemany(
                """
                insert into metric_summaries (
                    run_id, tag, first_step, last_step, first_value, last_value,
                    min_value, max_value, count, window_means_json, slope_last_points
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        metric.tag,
                        metric.first_step,
                        metric.last_step,
                        metric.first_value,
                        metric.last_value,
                        metric.min_value,
                        metric.max_value,
                        metric.count,
                        json.dumps(metric.window_means, sort_keys=True),
                        metric.slope_last_points,
                    )
                    for metric in run.metric_summaries
                ],
            )
            conn.execute("delete from metric_series where run_id = ?", (run.run_id,))
            conn.executemany(
                """
                insert into metric_series (
                    run_id, tag, points_json, original_count, sampled_count, first_step, last_step
                )
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        series.tag,
                        json.dumps(series.points),
                        series.original_count,
                        len(series.points),
                        _series_step(series.points, 0),
                        _series_step(series.points, -1),
                    )
                    for series in run.metric_series
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

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("select * from runs where run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._run_row_to_dict(row)

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

    def list_run_artifacts(self, run_id: str, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "select * from run_artifacts where run_id = ?"
        params: tuple[Any, ...] = (run_id,)
        if kind is not None:
            query += " and kind = ?"
            params = (run_id, kind)
        query += " order by kind, path"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "kind": row["kind"],
                "path": row["path"],
                "name": row["name"],
                "suffix": row["suffix"],
                "size_bytes": row["size_bytes"],
                "modified_time": row["modified_time"],
            }
            for row in rows
        ]

    def list_metric_summaries(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from metric_summaries where run_id = ? order by tag",
                (run_id,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "tag": row["tag"],
                "first_step": row["first_step"],
                "last_step": row["last_step"],
                "first_value": row["first_value"],
                "last_value": row["last_value"],
                "min_value": row["min_value"],
                "max_value": row["max_value"],
                "count": row["count"],
                "window_means": json.loads(row["window_means_json"]),
                "slope_last_points": row["slope_last_points"],
            }
            for row in rows
        ]

    def list_metric_series(self, run_id: str, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "select * from metric_series where run_id = ?"
        params: tuple[Any, ...] = (run_id,)
        if tag is not None:
            query += " and tag = ?"
            params = (run_id, tag)
        query += " order by tag"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "tag": row["tag"],
                "points": json.loads(row["points_json"]),
                "original_count": row["original_count"],
                "sampled_count": row["sampled_count"],
                "first_step": row["first_step"],
                "last_step": row["last_step"],
            }
            for row in rows
        ]

    def list_lineage(self, child_run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from lineage_edges where child_run_id = ? order by parent_run_id",
                (child_run_id,),
            ).fetchall()
        return [self._lineage_row_to_dict(row) for row in rows]

    def list_child_lineage(self, parent_run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from lineage_edges where parent_run_id = ? order by child_run_id",
                (parent_run_id,),
            ).fetchall()
        return [self._lineage_row_to_dict(row) for row in rows]

    def list_lineage_edges(self, project_name: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "select e.* from lineage_edges e"
        params: tuple[Any, ...] = ()
        if project_name is not None:
            query += " join runs child on child.run_id = e.child_run_id where child.project_name = ?"
            params = (project_name,)
        query += " order by e.parent_run_id, e.child_run_id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._lineage_row_to_dict(row) for row in rows]

    def upsert_run_observation(
        self,
        run_id: str,
        verdict: str,
        summary: str,
        tags: List[str],
        recommended_checkpoint: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into run_observations (
                    run_id, verdict, summary, tags_json, recommended_checkpoint, updated_at
                )
                values (?, ?, ?, ?, ?, ?)
                on conflict(run_id) do update set
                    verdict=excluded.verdict,
                    summary=excluded.summary,
                    tags_json=excluded.tags_json,
                    recommended_checkpoint=excluded.recommended_checkpoint,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    verdict,
                    summary,
                    json.dumps(tags),
                    recommended_checkpoint,
                    time.time(),
                ),
            )

    def get_run_observation(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("select * from run_observations where run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "verdict": row["verdict"],
            "summary": row["summary"],
            "tags": json.loads(row["tags_json"]),
            "recommended_checkpoint": row["recommended_checkpoint"],
            "updated_at": row["updated_at"],
        }

    def upsert_checkpoint_review(
        self,
        run_id: str,
        checkpoint: str,
        status: str,
        notes: str,
        tags: List[str],
        video_path: str = "",
        score: Optional[float] = None,
        recommended: bool = False,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                insert into checkpoint_reviews (
                    run_id, checkpoint, status, notes, tags_json, video_path,
                    score, recommended, updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(run_id, checkpoint) do update set
                    status=excluded.status,
                    notes=excluded.notes,
                    tags_json=excluded.tags_json,
                    video_path=excluded.video_path,
                    score=excluded.score,
                    recommended=excluded.recommended,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    checkpoint,
                    status,
                    notes,
                    json.dumps(tags),
                    video_path,
                    score,
                    int(recommended),
                    time.time(),
                ),
            )

    def list_checkpoint_reviews(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "select * from checkpoint_reviews where run_id = ? order by checkpoint",
                (run_id,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "checkpoint": row["checkpoint"],
                "status": row["status"],
                "notes": row["notes"],
                "tags": json.loads(row["tags_json"]),
                "video_path": row["video_path"],
                "score": row["score"],
                "recommended": bool(row["recommended"]),
                "updated_at": row["updated_at"],
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
            "git": json.loads(row["git_json"]),
            "latest_checkpoint": row["latest_checkpoint"],
            "parent_run_id": row["parent_run_id"],
            "parent_checkpoint": row["parent_checkpoint"],
        }

    def _project_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "name": row["name"],
            "local_cache_root": row["local_cache_root"],
            "parser_profile": row["parser_profile"],
            "preferred_metrics": json.loads(row["preferred_metrics_json"]),
            "log_patterns": json.loads(row["log_patterns_json"]),
            "tag_schema": json.loads(row["tag_schema_json"]),
        }

    def _lineage_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "parent_run_id": row["parent_run_id"],
            "child_run_id": row["child_run_id"],
            "relationship": row["relationship"],
            "parent_checkpoint": row["parent_checkpoint"],
            "intended_change": row["intended_change"],
            "note": row["note"],
            "confirmed": bool(row["confirmed"]),
        }


def _series_step(points: List[Dict[str, Any]], index: int) -> Optional[int]:
    if not points:
        return None
    return int(points[index]["step"])


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row["name"] for row in conn.execute(f"pragma table_info({table})")}
    if column not in existing:
        conn.execute(f"alter table {table} add column {column} {definition}")


def _run_artifacts(run: RunRecord) -> List[Dict[str, Any]]:
    records = []
    for kind, paths in (("video", run.videos), ("artifact", run.artifacts)):
        for path in paths:
            records.append(_artifact_record(Path(path), kind))
    return records


def _artifact_record(path: Path, kind: str) -> Dict[str, Any]:
    try:
        stat = path.stat()
        size_bytes = stat.st_size
        modified_time = stat.st_mtime
    except OSError:
        size_bytes = 0
        modified_time = 0.0
    return {
        "kind": kind,
        "path": str(path),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": size_bytes,
        "modified_time": modified_time,
    }
