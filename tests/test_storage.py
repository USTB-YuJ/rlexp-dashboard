import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.models import CheckpointRecord, LineageEdge, RunRecord
from rl_exp_dashboard.storage import DashboardStore


class DashboardStoreTests(unittest.TestCase):
    def test_store_persists_runs_checkpoints_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "dashboard.sqlite3"
            run_path = Path(tmp) / "logs" / "group" / "child"
            run_path.mkdir(parents=True)
            checkpoint_path = run_path / "model_100.pt"
            checkpoint_path.write_bytes(b"checkpoint")

            store = DashboardStore(db_path)
            store.initialize()
            store.upsert_project("unitree_rl_mjlab", local_cache_root=Path(tmp) / "cache")
            store.upsert_run(
                "unitree_rl_mjlab",
                RunRecord(
                    run_id="group/parent",
                    name="parent",
                    group="group",
                    path=run_path.parent / "parent",
                    modified_time=1.0,
                ),
            )
            store.upsert_run(
                "unitree_rl_mjlab",
                RunRecord(
                    run_id="group/child",
                    name="child",
                    group="group",
                    path=run_path,
                    modified_time=2.0,
                    params={"agent": {"algorithm": {"entropy_coef": 0.005}}},
                    checkpoints=[
                        CheckpointRecord(
                            path=checkpoint_path,
                            iteration=100,
                            size_bytes=10,
                            modified_time=3.0,
                            is_latest=True,
                        )
                    ],
                ),
            )
            store.upsert_lineage(
                LineageEdge(
                    parent_run_id="group/parent",
                    child_run_id="group/child",
                    relationship="finetune",
                    parent_checkpoint="model_50.pt",
                    intended_change="lower entropy",
                    note="Test child lineage.",
                )
            )

            runs = store.list_runs("unitree_rl_mjlab")
            child = next(run for run in runs if run["run_id"] == "group/child")
            checkpoints = store.list_checkpoints("group/child")
            lineage = store.list_lineage("group/child")

        self.assertEqual(child["latest_checkpoint"], "model_100.pt")
        self.assertEqual(child["params"]["agent"]["algorithm"]["entropy_coef"], 0.005)
        self.assertEqual(checkpoints[0]["iteration"], 100)
        self.assertEqual(checkpoints[0]["is_latest"], True)
        self.assertEqual(lineage[0]["parent_run_id"], "group/parent")
        self.assertEqual(lineage[0]["relationship"], "finetune")
        self.assertEqual(lineage[0]["intended_change"], "lower entropy")


if __name__ == "__main__":
    unittest.main()
