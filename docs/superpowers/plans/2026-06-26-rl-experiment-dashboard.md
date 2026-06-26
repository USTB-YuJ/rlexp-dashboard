# RL Experiment Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first usable local-first RL experiment dashboard foundation from the approved design: package skeleton, local run indexing, config diffing, SQLite metadata storage, API/CLI skeleton, and a minimal dashboard shell.

**Architecture:** Start with a Python package under `src/rl_exp_dashboard`. The first milestone keeps the parser/indexer/storage/test loop independent from heavy optional dependencies. FastAPI, TensorBoard, PyYAML, and the React frontend are integrated behind optional runtime boundaries so core behavior remains testable with the standard library.

**Tech Stack:** Python 3.9+, SQLite, argparse, unittest, optional FastAPI/uvicorn/PyYAML/tensorboard, later React/Vite.

---

## File Structure

- `pyproject.toml`: package metadata, console script, optional dependencies.
- `src/rl_exp_dashboard/__init__.py`: package version.
- `src/rl_exp_dashboard/models.py`: dataclasses for runs, checkpoints, params, metrics, lineage.
- `src/rl_exp_dashboard/config_diff.py`: flatten nested configs and compute structured diffs.
- `src/rl_exp_dashboard/structured_loader.py`: JSON/YAML loading with PyYAML when available and a small fallback parser for simple configs.
- `src/rl_exp_dashboard/indexer.py`: discover local training runs, params, checkpoints, videos, and event files.
- `src/rl_exp_dashboard/storage.py`: SQLite schema and repository methods.
- `src/rl_exp_dashboard/api.py`: FastAPI app factory, optional dependency.
- `src/rl_exp_dashboard/cli.py`: `rl-exp-dashboard` command.
- `src/rl_exp_dashboard/web_static/index.html`: minimal bundled dashboard shell for early smoke testing.
- `tests/test_config_diff.py`: tests config flattening/diff behavior.
- `tests/test_indexer.py`: tests local run discovery.
- `tests/test_storage.py`: tests SQLite persistence and lineage.
- `tests/test_cli.py`: tests CLI index command output.

## Task 1: Config Diff Core

**Files:**
- Create: `tests/test_config_diff.py`
- Create: `src/rl_exp_dashboard/config_diff.py`
- Create: `src/rl_exp_dashboard/models.py`

- [x] **Step 1: Write failing tests for flattening and diffing.**
- [x] **Step 2: Run tests and confirm import failure.**
- [x] **Step 3: Implement `flatten_config()` and `diff_configs()`.**
- [x] **Step 4: Run tests and confirm pass.**

## Task 2: Local Run Indexer

**Files:**
- Create: `tests/test_indexer.py`
- Create: `src/rl_exp_dashboard/structured_loader.py`
- Create: `src/rl_exp_dashboard/indexer.py`

- [x] **Step 1: Write failing tests using a synthetic RSL-RL-like log directory.**
- [x] **Step 2: Run tests and confirm missing indexer failure.**
- [x] **Step 3: Implement run discovery, params loading, checkpoint detection, event/video detection.**
- [x] **Step 4: Run tests and confirm pass.**

## Task 3: SQLite Storage and Lineage

**Files:**
- Create: `tests/test_storage.py`
- Create: `src/rl_exp_dashboard/storage.py`

- [x] **Step 1: Write failing tests for project/run/checkpoint persistence and lineage edge persistence.**
- [x] **Step 2: Run tests and confirm storage module failure.**
- [x] **Step 3: Implement schema initialization and repository methods.**
- [x] **Step 4: Run tests and confirm pass.**

## Task 4: CLI and Minimal API Shell

**Files:**
- Create: `tests/test_cli.py`
- Create: `src/rl_exp_dashboard/cli.py`
- Create: `src/rl_exp_dashboard/api.py`
- Create: `src/rl_exp_dashboard/web_static/index.html`
- Create: `src/rl_exp_dashboard/__init__.py`
- Create: `pyproject.toml`

- [x] **Step 1: Write failing tests for `rl-exp-dashboard index --log-root ... --db ...`.**
- [x] **Step 2: Run tests and confirm CLI is missing.**
- [x] **Step 3: Implement package metadata, CLI, optional FastAPI app factory, and static HTML shell.**
- [x] **Step 4: Run all tests and confirm pass.**

## Task 5: Verification and Commit

**Files:**
- All files above.

- [x] **Step 1: Run `python3 -m unittest discover -s tests`.**
- [x] **Step 2: Run `python3 -m compileall src tests`.**
- [x] **Step 3: Inspect `git diff --stat` and `git status --short`.**
- [x] **Step 4: Commit only dashboard implementation files.**

## Task 6: Metric Summary Pipeline

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/rl_exp_dashboard/metrics.py`
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `src/rl_exp_dashboard/api.py`

- [x] **Step 1: Write failing tests for scalar summary statistics.**
- [x] **Step 2: Implement metric summary dataclass and summary computation.**
- [x] **Step 3: Write failing tests for SQLite metric summary persistence.**
- [x] **Step 4: Implement `metric_summaries` storage table and repository method.**
- [x] **Step 5: Write failing tests for indexer metric reader injection.**
- [x] **Step 6: Implement optional TensorBoard event reader and connect it to `LocalRunIndexer`.**
- [x] **Step 7: Write failing tests for CLI metric summary persistence.**
- [x] **Step 8: Connect CLI indexing to metric summaries.**
- [x] **Step 9: Write failing tests for API metric summary payload helper.**
- [x] **Step 10: Add `/api/runs/{run_id}/metrics` payload support.**

## Task 7: Minimal Web Dashboard View

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Create: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing tests for run detail and compare payload helpers.**
- [x] **Step 2: Add storage helpers for single-run lookup and child lineage lookup.**
- [x] **Step 3: Implement run detail payload with run, checkpoints, metrics, and lineage.**
- [x] **Step 4: Implement compare payload with config diffs and metric last-value deltas.**
- [x] **Step 5: Write failing test for static dashboard structure and API calls.**
- [x] **Step 6: Replace placeholder static HTML with a run table and run detail panel.**
- [x] **Step 7: Add stable query-based API endpoints for run detail and metrics.**

## Task 8: Remote Sync Skeleton

**Files:**
- Create: `tests/test_sync.py`
- Create: `src/rl_exp_dashboard/sync.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_api.py`

- [x] **Step 1: Write failing tests for remote source sync plan generation.**
- [x] **Step 2: Implement `RemoteSource`, `SyncPlan`, and dry-run command builders for `rsync`, `scp`, and `ssh-tar`.**
- [x] **Step 3: Write failing tests for remote source and sync status persistence.**
- [x] **Step 4: Add `remote_sources` and `sync_status` SQLite tables and repository methods.**
- [x] **Step 5: Write failing tests for `rl-exp-dashboard sync --dry-run`.**
- [x] **Step 6: Implement CLI dry-run sync preview and sync status recording.**
- [x] **Step 7: Write failing tests for remote sources API payload.**
- [x] **Step 8: Add `/api/remote-sources` payload support.**

## Task 9: Manual Observation and Checkpoint Review

**Files:**
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing tests for run-level observation and checkpoint-level review persistence.**
- [x] **Step 2: Add `run_observations` and `checkpoint_reviews` SQLite tables and repository methods.**
- [x] **Step 3: Write failing tests for run detail payload and save helpers including observations/reviews.**
- [x] **Step 4: Add observation and checkpoint review fields to run detail payload.**
- [x] **Step 5: Add API helpers and POST routes for saving run observations and checkpoint reviews.**
- [x] **Step 6: Write failing static dashboard test for observation/review UI.**
- [x] **Step 7: Add manual observation and checkpoint review panels to the bundled dashboard shell.**

## Task 10: Sampled Metric Time-Series Charts

**Files:**
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/metrics.py`
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_indexer.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing tests for scalar series sampling, indexer injection, storage, API payload, and static chart hooks.**
- [x] **Step 2: Implement `MetricSeries`, scalar series downsampling, and optional TensorBoard series reading.**
- [x] **Step 3: Persist sampled series in SQLite and expose query helpers.**
- [x] **Step 4: Add metric series API payload/route.**
- [x] **Step 5: Add a minimal SVG trend chart to the bundled dashboard.**
- [x] **Step 6: Run targeted and full verification, then commit.**

## Task 11: Bundled Compare Runs UI

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing static dashboard tests for two-run compare controls and `/api/compare` usage.**
- [x] **Step 2: Add baseline/target selectors backed by indexed run IDs.**
- [x] **Step 3: Render config diffs and metric last-value deltas from the existing compare API.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 12: Actual Remote Sync Execution

**Files:**
- Modify: `src/rl_exp_dashboard/sync.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `tests/test_sync.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write failing tests for sync execution with an injectable subprocess runner and non-dry-run CLI status recording.**
- [x] **Step 2: Implement `execute_sync_plan()` with local directory creation and structured completion/failure results.**
- [x] **Step 3: Wire non-dry-run `rl-exp-dashboard sync` to execute the plan and record completed/failed status.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 13: Video and Artifact Persistence

**Files:**
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing tests for persisting indexed videos/artifacts, returning them in run detail, and showing them in the bundled dashboard.**
- [x] **Step 2: Add `run_artifacts` SQLite persistence with `kind=video/artifact`.**
- [x] **Step 3: Include artifacts in run detail payload.**
- [x] **Step 4: Render Videos and Artifacts sections in the bundled dashboard.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Later Milestones

- React/Vite frontend.
- Rich compare, lineage graph, checkpoint review UI.
- Packaged frontend build pipeline and Dockerfile.
