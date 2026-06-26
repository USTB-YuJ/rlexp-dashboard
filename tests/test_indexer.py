import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.indexer import LocalRunIndexer
from rl_exp_dashboard.structured_loader import load_structured_file


class LocalRunIndexerTests(unittest.TestCase):
    def test_load_structured_file_reads_simple_yaml_without_pyyaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.yaml"
            path.write_text(
                "\n".join(
                    [
                        "env:",
                        "  scene:",
                        "    num_envs: 2048",
                        "  rewards:",
                        "    alive:",
                        "      weight: 1.0",
                        "enabled: true",
                    ]
                ),
                encoding="utf-8",
            )

            data = load_structured_file(path)

        self.assertEqual(data["env"]["scene"]["num_envs"], 2048)
        self.assertEqual(data["env"]["rewards"]["alive"]["weight"], 1.0)
        self.assertEqual(data["enabled"], True)

    def test_discover_runs_indexes_params_checkpoints_events_and_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "g1_depth_parkour_amp" / "2026-06-25_15-19-19"
            params_dir = run_dir / "params"
            videos_dir = run_dir / "videos" / "play"
            params_dir.mkdir(parents=True)
            videos_dir.mkdir(parents=True)
            (params_dir / "env.yaml").write_text(
                "env:\n  scene:\n    num_envs: 2048\n  rewards:\n    alive:\n      weight: 1.0\n",
                encoding="utf-8",
            )
            (params_dir / "agent.yaml").write_text(
                "agent:\n  algorithm:\n    entropy_coef: 0.005\n",
                encoding="utf-8",
            )
            (run_dir / "events.out.tfevents.fake").write_text("event", encoding="utf-8")
            (run_dir / "model_100.pt").write_bytes(b"old")
            (run_dir / "model_250.pt").write_bytes(b"new")
            (videos_dir / "rl-video-step-0.mp4").write_bytes(b"video")

            runs = LocalRunIndexer(root).discover_runs()

        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual(run.run_id, "g1_depth_parkour_amp/2026-06-25_15-19-19")
        self.assertEqual(run.group, "g1_depth_parkour_amp")
        self.assertEqual(run.name, "2026-06-25_15-19-19")
        self.assertEqual(run.params["env"]["scene"]["num_envs"], 2048)
        self.assertEqual(run.params["agent"]["algorithm"]["entropy_coef"], 0.005)
        self.assertEqual([checkpoint.iteration for checkpoint in run.checkpoints], [100, 250])
        self.assertEqual(run.checkpoints[-1].path.name, "model_250.pt")
        self.assertTrue(run.checkpoints[-1].is_latest)
        self.assertEqual([path.name for path in run.event_files], ["events.out.tfevents.fake"])
        self.assertEqual([path.name for path in run.videos], ["rl-video-step-0.mp4"])


if __name__ == "__main__":
    unittest.main()
