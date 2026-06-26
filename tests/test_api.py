import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.api import metric_summaries_payload
from rl_exp_dashboard.models import MetricSummary, RunRecord
from rl_exp_dashboard.storage import DashboardStore


class ApiPayloadTests(unittest.TestCase):
    def test_metric_summaries_payload_returns_run_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "dashboard.sqlite3"
            store = DashboardStore(db_path)
            store.initialize()
            store.upsert_project("project", Path(tmp) / "cache")
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/run",
                    name="run",
                    group="group",
                    path=Path(tmp) / "run",
                    modified_time=1.0,
                    metric_summaries=[
                        MetricSummary(
                            tag="Train/mean_reward",
                            first_step=0,
                            last_step=10,
                            first_value=1.0,
                            last_value=2.0,
                            min_value=1.0,
                            max_value=2.0,
                            count=2,
                            window_means={"mean100": 1.5},
                            slope_last_points=0.1,
                        )
                    ],
                ),
            )

            payload = metric_summaries_payload(store, "group/run")

        self.assertEqual(payload["run_id"], "group/run")
        self.assertEqual(payload["metrics"][0]["tag"], "Train/mean_reward")
        self.assertEqual(payload["metrics"][0]["window_means"]["mean100"], 1.5)


if __name__ == "__main__":
    unittest.main()
