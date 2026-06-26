import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rl_exp_dashboard.cli import main
from rl_exp_dashboard.models import MetricSummary
from rl_exp_dashboard.storage import DashboardStore
from rl_exp_dashboard.sync import RemoteSource


class CliTests(unittest.TestCase):
    def test_index_command_indexes_local_log_root_into_sqlite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs" / "rsl_rl"
            run_dir = log_root / "g1_depth_parkour_amp" / "2026-06-25_15-19-19"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            (params_dir / "agent.yaml").write_text("agent:\n  seed: 7\n", encoding="utf-8")
            (run_dir / "model_10.pt").write_bytes(b"model")
            db_path = tmp_path / "dashboard.sqlite3"

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "index",
                        "--project",
                        "unitree_rl_mjlab",
                        "--log-root",
                        str(log_root),
                        "--db",
                        str(db_path),
                    ]
                )

            store = DashboardStore(db_path)
            runs = store.list_runs("unitree_rl_mjlab")

        self.assertEqual(exit_code, 0)
        self.assertIn("Indexed 1 runs", output.getvalue())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], "g1_depth_parkour_amp/2026-06-25_15-19-19")
        self.assertEqual(runs[0]["latest_checkpoint"], "model_10.pt")

    def test_index_command_can_store_metric_summaries_from_injected_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs" / "rsl_rl"
            run_dir = log_root / "group" / "run-with-metrics"
            run_dir.mkdir(parents=True)
            (run_dir / "events.out.tfevents.fake").write_text("event", encoding="utf-8")
            db_path = tmp_path / "dashboard.sqlite3"

            def metric_reader(_event_files):
                return [
                    MetricSummary(
                        tag="Train/mean_reward",
                        first_step=0,
                        last_step=20,
                        first_value=1.0,
                        last_value=3.0,
                        min_value=1.0,
                        max_value=3.0,
                        count=2,
                        window_means={"mean100": 2.0},
                        slope_last_points=0.1,
                    )
                ]

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "index",
                        "--project",
                        "unitree_rl_mjlab",
                        "--log-root",
                        str(log_root),
                        "--db",
                        str(db_path),
                    ],
                    metric_reader=metric_reader,
                )

            store = DashboardStore(db_path)
            metrics = store.list_metric_summaries("group/run-with-metrics")

        self.assertEqual(exit_code, 0)
        self.assertEqual(metrics[0]["tag"], "Train/mean_reward")
        self.assertEqual(metrics[0]["last_value"], 3.0)

    def test_index_command_can_use_imported_project_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "cache"
            run_dir = log_root / "group" / "run-from-config"
            run_dir.mkdir(parents=True)
            (run_dir / "model_20.pt").write_bytes(b"model")
            db_path = tmp_path / "dashboard.sqlite3"
            store = DashboardStore(db_path)
            store.initialize()
            store.upsert_project("unitree_rl_mjlab", log_root)

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "index",
                        "--project",
                        "unitree_rl_mjlab",
                        "--db",
                        str(db_path),
                    ]
                )

            runs = store.list_runs("unitree_rl_mjlab")

        self.assertEqual(exit_code, 0)
        self.assertEqual(runs[0]["run_id"], "group/run-from-config")

    def test_index_command_persists_inferred_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs" / "rsl_rl"
            parent = log_root / "group" / "baseline"
            child = log_root / "group" / "child"
            (parent / "params").mkdir(parents=True)
            (child / "params").mkdir(parents=True)
            (parent / "params" / "agent.yaml").write_text("agent:\n  seed: 1\n", encoding="utf-8")
            (child / "params" / "agent.yaml").write_text(
                "agent:\n  runner:\n    resume: true\n    load_run: baseline\n    load_checkpoint: model_100.pt\n",
                encoding="utf-8",
            )
            db_path = tmp_path / "dashboard.sqlite3"

            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "index",
                        "--project",
                        "unitree_rl_mjlab",
                        "--log-root",
                        str(log_root),
                        "--db",
                        str(db_path),
                    ]
                )

            store = DashboardStore(db_path)
            lineage = store.list_lineage("group/child")
            child_run = store.get_run("group/child")

        self.assertEqual(exit_code, 0)
        self.assertEqual(child_run["parent_run_id"], "group/baseline")
        self.assertEqual(lineage[0]["parent_run_id"], "group/baseline")
        self.assertEqual(lineage[0]["parent_checkpoint"], "model_100.pt")
        self.assertEqual(lineage[0]["relationship"], "resume")
        self.assertEqual(lineage[0]["confirmation_state"], "confirmed")
        self.assertEqual(lineage[0]["confidence_source"], "params")
        self.assertEqual(lineage[0]["confirmed"], True)

    def test_sync_dry_run_records_remote_source_and_prints_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "dashboard.sqlite3"
            cache_root = tmp_path / "cache"
            output = StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "sync",
                        "--project",
                        "unitree_rl_mjlab",
                        "--source-name",
                        "x-server",
                        "--host",
                        "example.com",
                        "--user",
                        "eai",
                        "--port",
                        "12188",
                        "--remote-log-root",
                        "/remote/logs/rsl_rl",
                        "--cache-root",
                        str(cache_root),
                        "--db",
                        str(db_path),
                        "--include-checkpoints",
                        "--dry-run",
                    ]
                )

            store = DashboardStore(db_path)
            sources = store.list_remote_sources("unitree_rl_mjlab")
            status = store.latest_sync_status("x-server", "unitree_rl_mjlab")

        self.assertEqual(exit_code, 0)
        self.assertIn("rsync", output.getvalue())
        self.assertIn("--dry-run", output.getvalue())
        self.assertIn("--include=model_*.pt", output.getvalue())
        self.assertEqual(sources[0]["name"], "x-server")
        self.assertEqual(status["status"], "dry-run")
        self.assertIn("--dry-run", status["command"])

    def test_sync_dry_run_can_use_imported_remote_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "dashboard.sqlite3"
            cache_root = tmp_path / "cache"
            store = DashboardStore(db_path)
            store.initialize()
            store.upsert_project("unitree_rl_mjlab", cache_root)
            store.upsert_remote_source(
                RemoteSource(
                    name="x-server",
                    host="example.com",
                    user="eai",
                    port=12188,
                    remote_log_root="/remote/logs/rsl_rl",
                    project="unitree_rl_mjlab",
                    method="rsync",
                    include_patterns=("params/***", "exports/***"),
                    exclude_patterns=("videos/***", "wandb/***"),
                )
            )
            output = StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "sync",
                        "--project",
                        "unitree_rl_mjlab",
                        "--source-name",
                        "x-server",
                        "--db",
                        str(db_path),
                        "--dry-run",
                    ]
                )

            status = store.latest_sync_status("x-server", "unitree_rl_mjlab")

        self.assertEqual(exit_code, 0)
        self.assertIn("eai@example.com:/remote/logs/rsl_rl/", output.getvalue())
        self.assertIn("--include=exports/***", output.getvalue())
        self.assertIn("--exclude=wandb/***", output.getvalue())
        self.assertEqual(status["local_path"], str(cache_root / "x-server" / "unitree_rl_mjlab" / "logs" / "rsl_rl"))

    def test_sync_command_executes_non_dry_run_with_injected_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "dashboard.sqlite3"
            cache_root = tmp_path / "cache"
            executed_commands = []

            class FakeCompletedProcess:
                returncode = 0
                stdout = "copied logs"
                stderr = ""

            def sync_runner(command, **kwargs):
                executed_commands.append((command, kwargs))
                return FakeCompletedProcess()

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "sync",
                        "--project",
                        "unitree_rl_mjlab",
                        "--source-name",
                        "x-server",
                        "--host",
                        "example.com",
                        "--user",
                        "eai",
                        "--port",
                        "12188",
                        "--remote-log-root",
                        "/remote/logs/rsl_rl",
                        "--cache-root",
                        str(cache_root),
                        "--db",
                        str(db_path),
                    ],
                    sync_runner=sync_runner,
                )

            store = DashboardStore(db_path)
            status = store.latest_sync_status("x-server", "unitree_rl_mjlab")

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(executed_commands), 1)
        self.assertNotIn("--dry-run", executed_commands[0][0])
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["message"], "copied logs")
        self.assertIn("Sync completed", output.getvalue())

    def test_project_import_command_persists_project_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "dashboard.sqlite3"
            config_path = tmp_path / "project.json"
            config_path.write_text(
                json.dumps(
                    {
                        "name": "unitree_rl_mjlab",
                        "local_cache_root": str(tmp_path / "cache"),
                        "parser_profile": "rsl_rl_tensorboard",
                        "preferred_metrics": ["Train/mean_reward"],
                        "log_patterns": ["logs/rsl_rl/*/*"],
                        "tag_schema": ["good-flat"],
                        "remote_sources": [
                            {
                                "name": "x-server",
                                "host": "example.com",
                                "user": "eai",
                                "port": 12188,
                                "remote_log_root": "/remote/logs/rsl_rl",
                                "method": "rsync",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "project",
                        "import",
                        "--config",
                        str(config_path),
                        "--db",
                        str(db_path),
                    ]
                )

            store = DashboardStore(db_path)
            project = store.get_project("unitree_rl_mjlab")
            sources = store.list_remote_sources("unitree_rl_mjlab")

        self.assertEqual(exit_code, 0)
        self.assertEqual(project["preferred_metrics"], ["Train/mean_reward"])
        self.assertEqual(sources[0]["name"], "x-server")
        self.assertIn("Imported project unitree_rl_mjlab", output.getvalue())


if __name__ == "__main__":
    unittest.main()
