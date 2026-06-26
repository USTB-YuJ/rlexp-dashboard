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

## Task 14: Params-Based Lineage Inference

**Files:**
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `tests/test_indexer.py`
- Modify: `tests/test_cli.py`

- [x] **Step 1: Write failing tests for inferring parent run/checkpoint from explicit resume/load-run params and persisting lineage during `index`.**
- [x] **Step 2: Implement conservative parent inference from nested `load_run` and `load_checkpoint`-style params.**
- [x] **Step 3: Persist inferred lineage edges during CLI indexing.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 15: Git Metadata Ingestion

**Files:**
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_indexer.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing tests for extracting explicit git metadata, persisting it with runs, and showing it in the bundled dashboard.**
- [x] **Step 2: Add `RunRecord.git_metadata` and conservative extraction from `git`-style params.**
- [x] **Step 3: Persist `git_json` on runs with a simple SQLite migration for existing databases.**
- [x] **Step 4: Render a Git Metadata panel in run detail.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 16: Safe Artifact Preview Serving

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing tests for whitelisted artifact file resolution and dashboard preview/link hooks.**
- [x] **Step 2: Add an API helper and `/api/artifact-file` route that only serves files already indexed for the requested run.**
- [x] **Step 3: Add video preview embeds and open links for indexed artifacts in the bundled dashboard.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 17: Run Table Filtering and Sorting

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing static dashboard tests for run search, project/group filters, sort controls, and filtered rendering hook.**
- [x] **Step 2: Add client-side run search, project filter, group filter, and modified/name sort controls.**
- [x] **Step 3: Wire controls to the indexed run list without changing the existing API.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 18: TOML Param File Support

**Files:**
- Modify: `src/rl_exp_dashboard/structured_loader.py`
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `tests/test_indexer.py`

- [x] **Step 1: Write failing tests for loading TOML params and discovering `.toml` files during run indexing.**
- [x] **Step 2: Add TOML loading via `tomllib`/`tomli` when available with a simple fallback parser for common scalar/table configs.**
- [x] **Step 3: Include `.toml` in local run param discovery.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 19: Remote Source Status Panel

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`

- [x] **Step 1: Write failing static dashboard tests for a remote source status panel, `/api/remote-sources` call, and render hook.**
- [x] **Step 2: Add a bundled dashboard panel listing configured remote sources and latest sync status.**
- [x] **Step 3: Load remote source status on page load and via refresh without changing existing APIs.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 20: Lineage Overview API

**Files:**
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for project-scoped lineage edge listing and a lineage overview payload.**
- [x] **Step 2: Add storage support for listing all lineage edges, optionally scoped by project.**
- [x] **Step 3: Add an API payload and route returning lineage nodes plus edges for graph-oriented UI work.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 21: Bundled Lineage Overview Panel

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing static dashboard tests for a lineage overview panel using `/api/lineage`.**
- [x] **Step 2: Add a bundled dashboard panel listing lineage edges and node counts.**
- [x] **Step 3: Load lineage overview on page load and expose a refresh control.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 22: Project Config Import

**Files:**
- Create: `src/rl_exp_dashboard/project_config.py`
- Create: `tests/test_project_config.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_api.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for loading a project config file and importing project/remote source/preferred metric metadata.**
- [x] **Step 2: Add project config parsing using existing structured config loading.**
- [x] **Step 3: Extend project storage and API payloads with parser profile, log patterns, tag schema, and preferred metrics.**
- [x] **Step 4: Add `project import --config ... --db ...` CLI command.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 23: Bundled Project Config Panel

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing static dashboard tests for a project config panel using `/api/projects`.**
- [x] **Step 2: Add a dashboard panel showing project cache roots, parser profiles, preferred metrics, and tag schemas.**
- [x] **Step 3: Load project metadata on page load and via refresh.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 24: CLI Uses Imported Project Config Defaults

**Files:**
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing CLI tests for `index` and `sync` reusing imported project config values.**
- [x] **Step 2: Make `index --log-root` optional when the project exists in SQLite.**
- [x] **Step 3: Make `sync` host/user/remote/cache args optional when a matching remote source exists in SQLite.**
- [x] **Step 4: Preserve explicit CLI args as overrides for imported config values.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 25: Remote Source Sync Pattern Persistence

**Files:**
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for persisting and displaying remote source include/exclude sync patterns.**
- [x] **Step 2: Add SQLite columns and migration for remote source include/exclude patterns.**
- [x] **Step 3: Preserve include/exclude patterns when CLI sync reuses an imported remote source.**
- [x] **Step 4: Render include/exclude patterns in the bundled Remote Sources panel.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 26: Manual Lineage Edge Editing

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for saving a manual lineage edge and showing a dashboard form.**
- [x] **Step 2: Add an API helper and `POST /api/lineage-edge` route backed by existing lineage storage.**
- [x] **Step 3: Add a bundled dashboard form for manually linking parent and child runs.**
- [x] **Step 4: Reload lineage overview after saving a manual edge.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 27: Run Table Summary Signals

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for run list review, media, lineage, and key metric summaries.**
- [x] **Step 2: Add a `runs_payload()` API helper that enriches run rows with table summary signals.**
- [x] **Step 3: Use the enriched payload in `/api/runs`.**
- [x] **Step 4: Render Reward, Review, Lineage, and Media columns in the bundled run table.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 28: Run Table Status Filters

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for observation/review flags and dashboard status filters.**
- [x] **Step 2: Add compact run flags for observations and reviewed checkpoints.**
- [x] **Step 3: Add Review and Flags controls to the bundled run table.**
- [x] **Step 4: Wire filters for reviewed, has video, has artifact, has parent, and has children.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 29: Experiment Timeline

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for a timeline payload and dashboard panel.**
- [x] **Step 2: Add `timeline_payload()` with chronological run, lineage, observation, and key metric fields.**
- [x] **Step 3: Add `/api/timeline` route.**
- [x] **Step 4: Render an Experiment Timeline panel in the bundled dashboard.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 30: Run Detail Params Browser

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing static dashboard tests for a Params Browser section and flatten helper.**
- [x] **Step 2: Add a Params Browser section to run detail.**
- [x] **Step 3: Render flattened dotted param paths and values from `run.params`.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 31: Grouped Compare Config Diffs

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for grouped compare diff payloads and dashboard hooks.**
- [x] **Step 2: Add backend grouping for reward, algorithm, observation/network, and curriculum/termination diffs.**
- [x] **Step 3: Render grouped compare diff panels before the full diff table.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 32: Run Detail Parent Comparison

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for parent comparison summary in run detail and dashboard hooks.**
- [x] **Step 2: Add `parent_compare` to run detail payload using lineage, grouped config diffs, and metric deltas.**
- [x] **Step 3: Render a Parent Comparison panel in run detail.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 33: Dashboard Project Creation

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for project save payloads and dashboard project form hooks.**
- [x] **Step 2: Add `save_project_payload()` and `/api/project` POST route.**
- [x] **Step 3: Render a Create / Update Project form in Project Configs and refresh after save.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 34: Dashboard Remote Source Creation

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for remote source save payloads and dashboard remote source form hooks.**
- [x] **Step 2: Add `save_remote_source_payload()` and `/api/remote-source` POST route.**
- [x] **Step 3: Render a Create / Update Remote Source form and refresh sources after save.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 35: Dashboard Project Indexing

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for project indexing payloads and dashboard indexing form hooks.**
- [x] **Step 2: Add `index_project_payload()` and `/api/index-project` POST route.**
- [x] **Step 3: Render an Index Project Logs form and refresh runs/timeline/lineage after indexing.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 36: Dashboard Remote Sync

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for remote sync payloads and dashboard sync form hooks.**
- [x] **Step 2: Add `sync_remote_source_payload()` and `/api/sync-remote-source` POST route.**
- [x] **Step 3: Render a Sync Remote Source form with dry-run default and refresh sync status.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 37: Lineage Edge State Metadata

**Files:**
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for lineage confirmation state, confidence source, result summary, and dashboard form hooks.**
- [x] **Step 2: Extend `LineageEdge`, SQLite schema migration, and lineage row serialization.**
- [x] **Step 3: Persist lineage metadata through `save_lineage_edge_payload()` and automatic resume lineage.**
- [x] **Step 4: Render lineage state/source/result fields in the bundled dashboard.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 38: CLI Inferred Lineage Metadata

**Files:**
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write a failing CLI test proving inferred resume lineage uses `confirmation_state=confirmed` and `confidence_source=params`.**
- [x] **Step 2: Add a shared `inferred_resume_lineage_edge()` helper and use it from API indexing.**
- [x] **Step 3: Use the same helper from CLI indexing.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 39: Package README Entry Point

**Files:**
- Create: `README.md`
- Modify: `pyproject.toml`
- Create: `tests/test_package_metadata.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests that `pyproject.toml` points to `README.md` and the README documents install, serve, sync, index, and Docker startup.**
- [x] **Step 2: Create a concise open-source README for local deployment and first-run workflow.**
- [x] **Step 3: Update `pyproject.toml` package metadata to use `README.md`.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 40: Default Server Install Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_package_metadata.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write a failing package metadata test proving plain `pip install rl-exp-dashboard` includes FastAPI and uvicorn.**
- [x] **Step 2: Move FastAPI and uvicorn into base package dependencies so `rl-exp-dashboard serve` works after normal install.**
- [x] **Step 3: Run targeted and full verification, then commit.**

## Task 41: Docker Deployment Skeleton

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Modify: `tests/test_package_metadata.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests proving Dockerfile exposes port 7860, installs the package, serves `/data`, and `.dockerignore` excludes local caches/logs.**
- [x] **Step 2: Add a minimal Python slim Dockerfile for the packaged dashboard server.**
- [x] **Step 3: Add `.dockerignore` entries for caches, logs, data downloads, videos, and VCS noise.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 42: Run Task And Algorithm Metadata

**Files:**
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_indexer.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for extracting, storing, returning, and rendering run `task_name` / `algorithm_name`.**
- [x] **Step 2: Extend `RunRecord`, indexer metadata extraction, and SQLite run schema migration.**
- [x] **Step 3: Surface task/algorithm fields through API payloads and bundled dashboard table/detail views.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 43: Run Start Time Metadata

**Files:**
- Modify: `src/rl_exp_dashboard/models.py`
- Modify: `src/rl_exp_dashboard/indexer.py`
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_indexer.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for parsing, storing, returning, sorting by, and rendering run `start_time`.**
- [x] **Step 2: Extend `RunRecord`, indexer timestamp parsing, and SQLite run schema migration.**
- [x] **Step 3: Use `start_time` in timeline payload ordering and dashboard table/detail/timeline display.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 44: Bundled Lineage Graph Visualization

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing static dashboard tests for a lineage SVG graph, render helper, and clickable run nodes.**
- [x] **Step 2: Render a compact bundled SVG graph from `/api/lineage` nodes and edges without new frontend dependencies.**
- [x] **Step 3: Wire graph nodes to run detail loading and keep the existing edge table/manual link form.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 45: Dashboard Project Scope Control

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing static dashboard tests for a global project scope selector and scoped API calls.**
- [x] **Step 2: Add a header-level project scope selector populated from `/api/projects`.**
- [x] **Step 3: Apply the selected project to runs, remote sources, lineage, and timeline fetches.**
- [x] **Step 4: Refresh scoped panels when the project scope changes and after project/index/sync actions.**
- [x] **Step 5: Run targeted and full verification, then commit.**

## Task 46: Rich Checkpoint Review UI

**Files:**
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing static dashboard tests for checkpoint score, recommended flag, and video path UI.**
- [x] **Step 2: Render score, recommended, and video path in checkpoint review rows.**
- [x] **Step 3: Add checkpoint review form controls for score and recommended status, preserving video path submission.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 47: Run Table Checkpoint Review Summary

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for run table payload checkpoint review summary fields and dashboard formatter hooks.**
- [x] **Step 2: Add recommended/best checkpoint review summary fields to `runs_payload()` rows.**
- [x] **Step 3: Include checkpoint review summaries in the bundled run table Review column and search text.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 48: Run Tag Summary And Filtering

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for run row observation/checkpoint tag summaries and dashboard tag filter hooks.**
- [x] **Step 2: Add `review_tags`, `checkpoint_review_tags`, and `all_tags` to run table payload rows.**
- [x] **Step 3: Render Tags in the bundled run table and add a client-side tag filter/search integration.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 49: Run Detail Reward And Termination Summaries

**Files:**
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for run detail reward/termination summary payloads and dashboard panels.**
- [x] **Step 2: Extract compact reward weights and termination entries from indexed params.**
- [x] **Step 3: Render Reward Summary and Termination Summary panels before the full params browser.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 50: Run Detail Config File Sources

**Files:**
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_indexer.py`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for indexed config file source persistence, API payload, and dashboard panel.**
- [x] **Step 2: Persist `RunRecord.param_files` as run config file metadata with migration-safe storage.**
- [x] **Step 3: Expose config files in run detail payload and render a Config Files panel before the params browser.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 51: Checkpoint Sync Opt-In

**Files:**
- Modify: `src/rl_exp_dashboard/sync.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_sync.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests proving rsync defaults exclude checkpoints and CLI/API/UI expose an opt-in checkpoint sync flag.**
- [x] **Step 2: Add `include_checkpoints` to sync plans and rsync include/exclude filters.**
- [x] **Step 3: Wire `--include-checkpoints` through CLI, API payloads, and the bundled remote sync form.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 52: Run Detail TensorBoard Event File Sources

**Files:**
- Modify: `src/rl_exp_dashboard/storage.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_storage.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for event file source persistence, API payload, and dashboard panel.**
- [x] **Step 2: Persist `RunRecord.event_files` as TensorBoard event source metadata.**
- [x] **Step 3: Expose event files in run detail payload and render a TensorBoard Event Files panel near metric charts.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Task 53: Single Run Markdown Experiment Report

**Files:**
- Create: `src/rl_exp_dashboard/report.py`
- Modify: `src/rl_exp_dashboard/api.py`
- Modify: `src/rl_exp_dashboard/cli.py`
- Modify: `src/rl_exp_dashboard/web_static/index.html`
- Modify: `tests/test_api.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_static_dashboard.py`
- Modify: `docs/superpowers/plans/2026-06-26-rl-experiment-dashboard.md`

- [x] **Step 1: Write failing tests for Markdown report generation, CLI export, and dashboard link hooks.**
- [x] **Step 2: Implement a report renderer that summarizes run metadata, lineage, metrics, manual observations, and checkpoint reviews.**
- [x] **Step 3: Expose the report through API, CLI, and run detail export link.**
- [x] **Step 4: Run targeted and full verification, then commit.**

## Later Milestones

- React/Vite frontend.
- Rich compare, lineage graph, checkpoint review UI.
- Packaged frontend build pipeline and Dockerfile.
