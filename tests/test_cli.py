import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from rl_exp_dashboard.cli import main
from rl_exp_dashboard.models import MetricSummary
from rl_exp_dashboard.storage import DashboardStore


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


if __name__ == "__main__":
    unittest.main()
