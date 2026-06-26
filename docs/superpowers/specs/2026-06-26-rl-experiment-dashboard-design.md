# RL Experiment Dashboard Design

## 1. Background

Robot reinforcement learning experiments accumulate many runs quickly. Each run may differ by reward weights, command logic, terrain curriculum, network architecture, AMP data, estimator losses, auxiliary heads, termination conditions, and training hyperparameters. If these changes are not recorded deliberately, it becomes difficult to answer basic questions later:

- What changed between two runs?
- Which parameter change improved the policy?
- Which checkpoint was visually best in play?
- Which run produced a bad behavior such as backward walking, hopping, sitting, arm shaking, or floating foothold predictions?
- Which model should be kept, replayed, exported, or discarded?

The goal is to build a local Web Dashboard for robot RL experiment management. It should automatically index logs and parameters, compare run configuration differences, summarize metrics, and provide structured manual notes for policy behavior.

## 2. Product Goal

Build a local-first dashboard that turns scattered RL training logs into a searchable experiment notebook.

The first production-quality version should let a user:

1. Sync experiment logs from remote training machines into a local cache.
2. Index each run's parameters, checkpoints, TensorBoard metrics, git metadata, videos, and artifacts.
3. Compare multiple runs and show config diffs, metric trends, and result summaries.
4. Record human observations at both run level and checkpoint level.
5. Launch from a single command and remain easy to install on new machines.

## 3. Target Users

Primary user:

- Robot learning researcher or engineer running many IsaacLab/MuJoCo/RSL-RL style experiments.
- Frequently adjusts reward terms, curriculum logic, command logic, AMP data, network architecture, and estimator heads.
- Needs to understand experiment history after days or weeks of iteration.

Secondary user:

- A lab teammate who wants to browse previous experiments, inspect configs, compare checkpoints, and add play observations.
- An external open-source user who wants to adapt the tool to their own RL log directory.

## 4. Delivery Form

The tool should be built as a local Web Dashboard with Python backend and packaged frontend.

Recommended distribution:

- Python package:
  - Install: `pip install rl-exp-dashboard`
  - Start: `rl-exp-dashboard serve --workspace ~/rl-exp-dashboard`
- Optional editable developer install:
  - `pip install -e .`
  - `pnpm install`
  - `pnpm dev`
- Optional Docker image:
  - `docker run -p 7860:7860 -v ~/rl-exp-dashboard:/data rl-exp-dashboard`

The frontend should be built during package release and bundled as static assets inside the Python package. End users should not need Node.js for normal use.

## 5. Architecture

Use a local-first architecture:

```text
Remote training hosts
        |
        | SSH/SCP or rsync sync
        v
Local cache directory
        |
        | indexer parses logs, params, events, artifacts
        v
SQLite metadata database
        |
        | FastAPI API
        v
React/Vite Web Dashboard
```

### 5.1 Backend

Use Python + FastAPI.

Responsibilities:

- Manage project/workspace configuration.
- Sync remote logs into local cache.
- Parse run directories.
- Extract YAML/JSON/TOML params.
- Read TensorBoard event scalar summaries.
- Detect checkpoints, videos, exported policies, and other artifacts.
- Compute config diffs between runs.
- Store indexed metadata and user annotations in SQLite.
- Serve API endpoints and packaged frontend static files.

Python is preferred because RL logs, TensorBoard event files, YAML configs, SSH workflows, and robot training scripts are already Python-centered.

### 5.2 Frontend

Use React + Vite.

Responsibilities:

- Run table with filtering, sorting, and grouping.
- Run detail page.
- Multi-run config diff view.
- Metric trend charts.
- Manual observation editor.
- Checkpoint review panel.
- Video/artifact preview.
- Remote sync status UI.

React is preferred over Streamlit because this tool should become a polished open-source dashboard, not just a script panel.

### 5.3 Storage

Use SQLite as the local metadata database.

SQLite stores:

- Projects.
- Remote hosts.
- Run metadata.
- Run lineage edges.
- Parsed params.
- Metric summaries.
- Checkpoints.
- Artifacts.
- Manual observations.
- Tags.
- Comparison groups.
- Sync status.

Raw logs and artifacts stay in the local cache directory. The database stores paths and normalized summaries, not copies of every large file.

## 6. Core Concepts

### 6.1 Project

A project is a logical training family, such as:

- `unitree_rl_mjlab`
- `g1_depth_parkour_amp`
- `h1_dwaq_payload`

Each project has:

- Name.
- Local cache root.
- Optional remote sources.
- Log path patterns.
- Parser profile.
- Preferred metrics.
- Tag schema.

### 6.2 Remote Source

A remote source describes where logs come from.

Fields:

- Display name, such as `x-server`.
- SSH host, port, user.
- Remote log root, such as `/home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl`.
- Include/exclude patterns.
- Sync method: `rsync`, `scp`, or `ssh tar`.
- Last sync timestamp.

Passwords should not be stored in plaintext. First version can rely on SSH keys or the user's local SSH agent. If password login is needed, prompt per sync session and avoid saving it.

### 6.3 Run

A run is one training directory, such as:

```text
logs/rsl_rl/g1_depth_parkour_amp/2026-06-25_15-19-19
```

Indexed run fields:

- Run ID.
- Project.
- Task name.
- Algorithm name.
- Start time.
- Last modified time.
- Host/source.
- Git commit and branch if available.
- Dirty diff snapshot if available.
- Parent run if known.
- Parent checkpoint if known.
- Number of iterations.
- Latest checkpoint.
- Best checkpoint candidates.
- Params files.
- TensorBoard event files.
- Videos.
- Exported policies.
- User tags.
- Human run summary.

### 6.4 Run Lineage

Run lineage records how experiments are connected. This is important because robot RL experiments often form chains:

```text
baseline run
  -> lower entropy run
    -> foothold patch head run
      -> reward tuning run
```

Lineage fields:

- Child run.
- Parent run.
- Parent checkpoint when training was resumed from a model file.
- Relationship type: `resume`, `finetune`, `ablation`, `rerun`, `manual-link`.
- Main intended change.
- Automatically detected config diff against the parent.
- Human note explaining why this child run was created.

The dashboard should infer lineage when possible:

- If params contain resume/load-run metadata, use it.
- If a run directory or command includes a checkpoint path, parse it.
- If git metadata and timestamps suggest a likely parent, show it as a suggestion, not as a confirmed link.
- Always allow manual correction.

Lineage creation should be an explicit workflow, not only an inferred background detail:

1. When a run is indexed, the system creates a confirmed edge only from high-confidence evidence such as `load_run`, `resume_run`, or a concrete checkpoint path in saved params.
2. When a user reviews experiments, the dashboard exposes a manual link form to connect parent and child runs, choose the relationship type, record the parent checkpoint, and write the intended change.
3. When a future "start from this run" workflow exists, the dashboard should pre-fill parent run, checkpoint, git commit, and intended-change notes before launching or documenting the child run.
4. Suggested lineage links should remain unconfirmed until the user accepts them, so speculative timestamp/git heuristics do not pollute the experiment story.

Lineage should make it possible to answer:

- Which run did this one come from?
- Which checkpoint was used as the parent?
- What changed from parent to child?
- Did that change improve or hurt the result?
- Which branch of experiments should be continued or abandoned?

Run lineage should be treated as a first-class experiment story, not just as a display field. In robot RL, the most important comparison is often not chronological order, but "what did I change from the previous meaningful baseline?" A run may be started from a checkpoint, copied from a previous config, or manually recreated after several unrelated experiments. The dashboard should preserve that relationship explicitly.

Recommended lineage state model:

- `confirmed`: backed by high-confidence resume metadata or manually accepted by the user.
- `suggested`: inferred from weak signals such as nearby timestamps, similar configs, same git commit, or naming conventions.
- `rejected`: a suggestion that the user dismissed, so the system does not keep proposing it.

Each lineage edge should store:

- Parent run ID.
- Child run ID.
- Parent checkpoint, when available.
- Relationship type.
- Confidence source: `params`, `checkpoint-path`, `manual`, `git-timestamp-suggestion`, `name-pattern`.
- Confirmation state.
- Intended change summary.
- Optional result summary, such as "reward improved but gait became high-frequency".
- Created/updated timestamp.

Lineage establishment should happen through several paths:

1. Automatic confirmed edges from explicit training metadata:
   - `resume: true`
   - `load_run`
   - `load_checkpoint`
   - absolute or relative checkpoint paths saved in params.
2. Automatic suggested edges from weaker evidence:
   - same task/group and nearby start time.
   - child config differs by only a few keys from a recent run.
   - child run uses the same git commit or the next commit after parent.
   - run name contains a parent or experiment-series hint.
3. Manual linking in the dashboard:
   - pick parent and child.
   - choose relationship type.
   - enter parent checkpoint and intended change.
   - accept/reject suggested links.
4. Future launch integration:
   - "Continue from this checkpoint" should pre-fill parent metadata before training starts.
   - The generated run note should include intended change, expected metric to improve, and risk being tested.

Lineage-aware comparison should become the default analysis mode:

- A child run detail page should compare against its confirmed parent by default.
- The compare page should highlight config changes grouped by reward, command, terrain, curriculum, network, AMP, estimator, and algorithm.
- Result deltas should sit next to config deltas, so the user sees both "what changed" and "what happened".
- If a child has worse metrics but better play behavior, manual notes should make that visible in the same parent-child comparison.
- If several children share one parent, the dashboard should show a branch comparison table to answer which ablation is worth continuing.

### 6.5 Checkpoint Review

Some behaviors can only be judged in play. The dashboard should support checkpoint-level observations.

Fields:

- Checkpoint file.
- Iteration.
- Evaluation status: `unreviewed`, `good`, `mixed`, `bad`, `exported`.
- User notes.
- Behavior tags.
- Linked videos.
- Recommended for play/export flag.
- Optional numeric score.

This avoids losing details such as "model_11300.pt walks well on rough terrain but sits on stairs occasionally".

### 6.6 Manual Observation

Manual observation exists at two levels:

Run-level observation:

- Overall result.
- Main behavior.
- Known failure modes.
- Whether to keep, continue, compare, or discard.
- Links to useful checkpoints.

Checkpoint-level observation:

- Play behavior for a specific checkpoint.
- Terrain-specific comments.
- Video link.
- Deployment/export decision.

Suggested behavior tags:

- `forward-walk`
- `backward-walk`
- `hopping`
- `sitting`
- `falling`
- `arm-shaking`
- `high-frequency-gait`
- `good-stairs`
- `bad-stairs`
- `good-rough`
- `foothold-floating`
- `foothold-good`
- `needs-retrain`
- `candidate`
- `exported`

Users should be able to add custom tags.

## 7. Data Sources

### 7.1 Params

The indexer should parse common parameter files:

- `params/env.yaml`
- `params/agent.yaml`
- `params/*.yaml`
- `config.yaml`
- `hydra` output if present.
- JSON/TOML files when available.

The first parser profile should support the current RSL-RL style logs used in `unitree_rl_mjlab`.

Important extracted categories:

- Task/env name.
- Number of envs.
- Terrain configuration.
- Command configuration.
- Reward terms and weights.
- Termination terms.
- Curriculum terms.
- Observation groups and dimensions.
- Network architecture.
- Algorithm hyperparameters.
- AMP configuration.
- Estimator/auxiliary loss coefficients.
- Depth sensor settings.
- Motion data path.

### 7.2 TensorBoard Scalars

The backend should read TensorBoard event files and store summarized scalar metrics.

For each scalar tag:

- First value.
- Last value.
- Min/max.
- Mean over recent windows.
- Step of last value.
- Optional slope over recent points.

Default recent windows:

- Last 100 steps.
- Last 500 steps.
- Last 1000 steps.
- Last 5000 steps.

The raw event file remains in cache. The database stores summaries and enough sampled series for charts.

### 7.3 Checkpoints

Detect checkpoint files such as:

- `model_0.pt`
- `model_11300.pt`
- `model_best.pt`

Extract:

- Iteration number.
- File size.
- Modified time.
- Whether it is latest.
- Whether it has user review.
- Linked videos or exported artifacts.

### 7.4 Videos and Artifacts

Detect:

- Play videos.
- Foothold visualization videos.
- Exported ONNX/JIT policies.
- Config snapshots.
- Git snapshots.
- Evaluation outputs.

Videos should be playable in the dashboard when browser-supported.

## 8. Main Views

### 8.1 Home / Project Overview

Shows:

- Projects.
- Total runs.
- Last sync status.
- Recent runs.
- Runs needing review.
- Candidate checkpoints.

Primary actions:

- Add project.
- Sync remote.
- Open latest run.
- Compare selected runs.

### 8.2 Run Table

A dense table for scanning many experiments.

Columns:

- Run name/time.
- Task.
- Parent run.
- Git commit/branch.
- Latest checkpoint.
- Last reward.
- Recent reward mean.
- Episode length.
- Success rate.
- Terrain level.
- Termination highlights.
- Key tags.
- Human verdict.

Filters:

- Project.
- Date range.
- Task name.
- Tags.
- Reward threshold.
- Success threshold.
- Has notes.
- Has video.
- Has reviewed checkpoint.
- Has parent.
- Has children.
- Git commit.

### 8.3 Run Detail

Sections:

- Run summary.
- Lineage summary.
- Human notes.
- Key metrics.
- Reward terms.
- Terminations.
- Curriculum.
- Checkpoints.
- Videos/artifacts.
- Params browser.
- Git metadata.

This page should answer: "What happened in this run?"

The lineage summary should show:

- Parent run and parent checkpoint if known.
- Direct child runs.
- The main config changes from parent to current run.
- Human note describing the experiment intent.
- A quick verdict comparing current run against parent.

### 8.4 Compare Runs

Compare two or more runs.

Panels:

- Metric comparison chart.
- Config diff.
- Reward weight diff.
- Algorithm hyperparameter diff.
- Observation/network diff.
- Curriculum/termination diff.
- Human verdict comparison.
- Parent-child improvement summary when comparing lineage-connected runs.

The config diff should be structured rather than only text-based. For example:

```text
rewards.action_rate_l2.weight
  run A: -1e-4
  run B: -1e-3

algorithm.entropy_coef
  run A: 0.01
  run B: 0.005
```

The first version can use tree paths generated from parsed YAML.

### 8.5 Experiment Timeline

A chronological timeline of runs.

Each entry shows:

- Run time.
- Main changed params compared to previous selected baseline.
- Result summary.
- Human verdict.

This is useful for reconstructing the story of an experiment series.

### 8.6 Lineage Graph

The lineage graph shows experiment ancestry.

It should support:

- Nodes as runs.
- Edges as parent-child relationships.
- Edge labels such as `resume`, `finetune`, `ablation`, or `rerun`.
- Node color by human verdict or key metric.
- Click a node to open run detail.
- Click an edge to show parent-child config diff and result delta.

The first implementation can be a simple tree/list view if a graph library would slow down V0. The data model should still support graph rendering later.

### 8.7 Checkpoint Review

For each run:

- List checkpoints.
- Show nearest metric values at checkpoint iteration.
- Attach play video.
- Add behavior tags.
- Mark recommended checkpoint.

This is where human visual inspection enters the system.

## 9. Diff Design

Parameter diff is one of the core features.

The backend should flatten nested configs into dotted paths:

```text
env.rewards.action_rate_l2.weight = -0.001
agent.algorithm.entropy_coef = 0.005
env.observations.actor.depth.crop_width = 48
```

Diff categories:

- Added key.
- Removed key.
- Changed value.
- Type changed.

The UI should support:

- Show all diffs.
- Hide unchanged.
- Filter by category prefix, such as `rewards`, `algorithm`, `observations`.
- Mark important diffs.
- Collapse noisy paths.
- Compare against an explicit parent run when lineage is available.
- Show result delta next to important config deltas, such as reward, episode length, success rate, and termination changes.

Config values should preserve source file and path for traceability.

When a run has a parent, the default diff baseline should be the parent run. Manual compare should still allow any two runs to be selected.

## 10. Result Summary Design

Automatic result summaries should be configurable per project.

For the current parkour project, useful default metrics include:

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Metrics/twist/target_episode_success`
- `Metrics/twist/target_reached_count`
- `Curriculum/terrain_levels`
- `Curriculum/terrain_levels_by_type/*`
- `Episode_Metrics/harness_stage_fraction`
- `Episode_Metrics/harness_force_norm`
- `Episode_Termination/body_height`
- `Episode_Termination/fell_over`
- `Episode_Termination/time_out`
- `Metrics/foothold_center_xy_mae`
- `Metrics/foothold_center_h_mae`
- `Metrics/foothold_box_xy_mae`
- `Metrics/foothold_yaw_error`
- `Metrics/estimator_velocity_mae`
- `Metrics/estimator_contact_accuracy`

The dashboard should not hard-code only these tags. Project config should allow preferred metric presets.

## 11. Sync Design

### 11.1 Local Cache

Each remote source syncs into a local cache directory:

```text
~/rl-exp-dashboard/cache/
  x-server/
    unitree_rl_mjlab/
      logs/
        rsl_rl/
          g1_depth_parkour_amp/
            2026-06-25_15-19-19/
```

The database stores canonical local paths.

### 11.2 Remote Sync Methods

Preferred first implementation:

- `rsync` over SSH when available.

Fallbacks:

- `scp` for simple copying.
- `ssh tar` streaming for environments where rsync is unavailable.

Sync should support:

- Dry run preview.
- Include/exclude patterns.
- Skip large checkpoint files unless requested.
- Sync latest N checkpoints.
- Sync event files and params by default.
- Sync videos on demand.

Suggested default:

- Always sync params and event files.
- Sync latest checkpoint.
- Sync checkpoints explicitly marked by user.
- Sync videos when user requests or when small enough.

This avoids copying hundreds of checkpoint files unnecessarily.

### 11.3 Sync Safety

Do not delete local cached logs by default.

If a remote file disappears:

- Mark local file as orphaned.
- Keep it unless user runs cleanup.

Avoid storing remote passwords. Use SSH key or agent-based authentication as the normal path.

## 12. Open-Source Readiness

The tool should be useful outside the current project.

Design choices for open source:

- Parser profiles instead of hard-coded project logic.
- Configurable metric presets.
- Configurable log path patterns.
- Generic YAML diff engine.
- Plugin-like parser hooks later.
- Simple Python package install.
- Optional Docker image.
- Clear sample dataset or anonymized demo logs.

Initial parser profiles:

- `rsl_rl_tensorboard`
- `isaaclab_rsl_rl`
- `generic_tensorboard`

Current `unitree_rl_mjlab` support can be implemented as a project preset built on top of the generic RSL-RL profile.

## 13. First Version Scope

Version 0 should include:

- Local workspace creation.
- Add local log directory.
- Add remote source.
- Manual remote sync.
- Parse run directories.
- Parse params YAML.
- Parse TensorBoard scalar summaries.
- Detect checkpoints.
- Detect videos.
- SQLite cache.
- Run table.
- Run detail page.
- Parent run and parent checkpoint metadata.
- Manual run lineage editing.
- Two-run config diff.
- Run-level notes and tags.
- Checkpoint-level notes and tags.
- Packaged one-command local server.

### 13.1 Version 0 User Flow

The minimum useful workflow should be:

1. User installs the package and starts the server:

   ```bash
   rl-exp-dashboard serve --workspace ~/rl-exp-dashboard
   ```

2. The browser opens the local dashboard.
3. User creates a project named `unitree_rl_mjlab`.
4. User adds either:
   - A local log root, such as `~/logs/rsl_rl`.
   - A remote source, such as `eai@x-server:/home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl`.
5. User clicks `Sync` or `Index`.
6. Dashboard lists discovered runs.
7. User opens a run and sees:
   - Latest checkpoint.
   - Parent run and child runs when known.
   - Main metric summaries.
   - Reward weights.
   - Termination summaries.
   - Config files.
8. User selects two runs and opens `Compare`.
9. Dashboard shows config diffs and metric comparison.
10. User links a run to its parent if automatic inference missed it.
11. User writes run-level notes and checkpoint-level play observations.

This flow should work without writing a custom parser. Project presets can improve naming and metric defaults, but the generic TensorBoard/YAML parser should still provide a useful baseline.

Version 0 should not include:

- Multi-user authentication.
- Cloud sync.
- Real-time training process control.
- Automatic policy scoring from videos.
- Automatic behavior recognition.
- Distributed database.
- Full TensorBoard replacement.

## 14. Future Extensions

Useful later features:

- Automated play/eval job launcher.
- Side-by-side video comparison.
- Metric anomaly detection.
- "What changed before this improvement?" assistant.
- Git diff ingestion and display.
- Reward-term contribution analysis.
- Rich lineage analytics across experiment branches.
- Model registry export status.
- Deployment checklist.
- Team comments and review workflow.
- Optional LLM-assisted run summary generation.

## 15. Risks and Mitigations

Risk: Logs are not standardized across projects.

Mitigation: Use parser profiles and project presets. Keep the core data model generic.

Risk: Remote sync copies too much data.

Mitigation: Sync params/events by default, latest checkpoint only, and videos/checkpoints on demand.

Risk: Manual note-taking becomes burdensome.

Mitigation: Make notes optional, tag-based, and quick to fill. Provide run-level notes first and checkpoint notes as expandable detail.

Risk: TensorBoard event parsing is slow for large logs.

Mitigation: Cache summaries in SQLite and only re-parse files whose size or modified time changed.

Risk: Automatic lineage inference links runs incorrectly.

Mitigation: Treat inferred lineage as suggested until confirmed when confidence is low. Always allow manual relinking and store whether a lineage edge was inferred or user-confirmed.

Risk: Dashboard becomes too project-specific.

Mitigation: Keep `unitree_rl_mjlab` as a preset, not the core architecture.

Risk: Users expect training control.

Mitigation: Clearly position V0 as experiment management, not training orchestration.

## 16. Suggested Development Milestones

Milestone 1: Local indexer

- Point tool at a local log directory.
- Parse runs, params, event summaries, checkpoints.
- Store in SQLite.

Milestone 2: Basic dashboard

- Run table.
- Run detail.
- Manual notes.
- Checkpoint notes.
- Manual parent-child run linking.

Milestone 3: Config diff and comparison

- Flatten YAML configs.
- Two-run diff.
- Parent-child diff.
- Metric comparison.

Milestone 4: Remote sync

- Add SSH remote.
- Sync params/events/latest checkpoint.
- Track sync status.

Milestone 5: Packaging

- Bundle frontend.
- Add CLI.
- Add Dockerfile.
- Add sample/demo docs.

## 17. Current Decisions

Confirmed decisions:

- The product should be a local Web Dashboard.
- It should use local cache plus remote sync.
- Human observations should exist at both run level and checkpoint level.
- Run lineage should be a core data model, including parent run and parent checkpoint relationships.
- The implementation should be open-source friendly.
- The preferred architecture is FastAPI backend plus React/Vite frontend.
- Normal users should be able to install and start it with a Python package command.
- Docker should be supported as an optional deployment path.

Design choices still intended for later refinement:

- Exact UI visual layout.
- Exact project config file format.
- Whether first implementation lives inside the current robot repo or as a new standalone repo.
- Which remote sync method is the default on machines without `rsync`.
- How much metric time-series data to keep in SQLite versus reading from event files on demand.
