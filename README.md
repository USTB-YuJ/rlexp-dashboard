# RL Experiment Dashboard

Local-first Web Dashboard for robot reinforcement learning experiment logs.

The dashboard indexes RSL-RL style training folders, stores summaries in SQLite,
compares run parameters, tracks parent-child run lineage, and keeps human play
observations close to checkpoints and videos.

## Install

```bash
pip install rl-exp-dashboard
```

For local development:

```bash
pip install -e .[dev]
```

## Start The Dashboard

```bash
rl-exp-dashboard serve --workspace ~/rl-exp-dashboard
```

Open `http://127.0.0.1:7860` in your browser. The workspace stores
`dashboard.sqlite3` plus any local cache data you choose to sync or index.

## First Run Workflow

Create or import a project, then either index local logs or sync remote logs.

Index an existing local log root:

```bash
rl-exp-dashboard index \
  --project unitree_rl_mjlab \
  --log-root ~/workspace/unitree_rl_mjlab/logs/rsl_rl \
  --db ~/rl-exp-dashboard/dashboard.sqlite3
```

Preview a remote sync without copying files:

```bash
rl-exp-dashboard sync \
  --project unitree_rl_mjlab \
  --source-name x-server \
  --host example.com \
  --user eai \
  --port 12188 \
  --remote-log-root /home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl \
  --cache-root ~/rl-exp-dashboard/cache \
  --db ~/rl-exp-dashboard/dashboard.sqlite3 \
  --dry-run
```

Remove `--dry-run` to execute the sync. Remote passwords are not stored; use SSH
keys or your local SSH agent.

## Project Config

You can import reusable project configuration:

```bash
rl-exp-dashboard project import \
  --config dashboard-project.yaml \
  --db ~/rl-exp-dashboard/dashboard.sqlite3
```

Project configs can define the local cache root, parser profile, preferred
metrics, tag schema, and remote sources.

## Docker

The package is designed to run locally or in a small container:

```bash
docker run -p 7860:7860 -v ~/rl-exp-dashboard:/data rl-exp-dashboard
```

## What Gets Indexed

- YAML, JSON, and TOML parameter files.
- TensorBoard scalar summaries and sampled metric series.
- Checkpoints such as `model_1000.pt`.
- Videos and exported policy artifacts.
- Git metadata snapshots when available.
- Run lineage inferred from resume/load-run parameters.
- Manual run observations and checkpoint reviews.

## Design Notes

The design document lives at
`docs/superpowers/specs/2026-06-26-rl-experiment-dashboard-design.md`.
