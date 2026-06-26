import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.indexer import LocalRunIndexer
from rl_exp_dashboard.models import MetricSeries, MetricSummary
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

    def test_load_structured_file_reads_toml_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent.toml"
            path.write_text(
                "\n".join(
                    [
                        "[agent.algorithm]",
                        "entropy_coef = 0.005",
                        "use_amp = true",
                        "tags = [\"baseline\", \"parkour\"]",
                        "",
                        "[agent.runner]",
                        "load_run = \"baseline\"",
                    ]
                ),
                encoding="utf-8",
            )

            data = load_structured_file(path)

        self.assertEqual(data["agent"]["algorithm"]["entropy_coef"], 0.005)
        self.assertEqual(data["agent"]["algorithm"]["use_amp"], True)
        self.assertEqual(data["agent"]["algorithm"]["tags"], ["baseline", "parkour"])
        self.assertEqual(data["agent"]["runner"]["load_run"], "baseline")

    def test_discover_runs_indexes_params_checkpoints_events_and_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "g1_depth_parkour_amp" / "2026-06-25_15-19-19"
            params_dir = run_dir / "params"
            videos_dir = run_dir / "videos" / "play"
            params_dir.mkdir(parents=True)
            videos_dir.mkdir(parents=True)
            (params_dir / "env.yaml").write_text(
                "env:\n  task:\n    name: Unitree-G1-Depth-Parkour\n  scene:\n    num_envs: 2048\n  rewards:\n    alive:\n      weight: 1.0\n",
                encoding="utf-8",
            )
            (params_dir / "agent.yaml").write_text(
                "agent:\n  algorithm:\n    name: rsl_rl_ppo\n    entropy_coef: 0.005\n",
                encoding="utf-8",
            )
            (params_dir / "runner.toml").write_text(
                "[runner]\nexperiment_name = \"parkour\"\nmax_iterations = 1000\n",
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
        self.assertEqual(run.task_name, "Unitree-G1-Depth-Parkour")
        self.assertEqual(run.algorithm_name, "rsl_rl_ppo")
        self.assertEqual(run.params["env"]["scene"]["num_envs"], 2048)
        self.assertEqual(run.params["agent"]["algorithm"]["entropy_coef"], 0.005)
        self.assertEqual(run.params["runner"]["experiment_name"], "parkour")
        self.assertEqual(run.params["runner"]["max_iterations"], 1000)
        self.assertEqual([checkpoint.iteration for checkpoint in run.checkpoints], [100, 250])
        self.assertEqual(run.checkpoints[-1].path.name, "model_250.pt")
        self.assertTrue(run.checkpoints[-1].is_latest)
        self.assertEqual([path.name for path in run.event_files], ["events.out.tfevents.fake"])
        self.assertEqual([path.name for path in run.videos], ["rl-video-step-0.mp4"])

    def test_discover_runs_uses_metric_reader_for_event_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "group" / "run-a"
            run_dir.mkdir(parents=True)
            event_file = run_dir / "events.out.tfevents.fake"
            event_file.write_text("event", encoding="utf-8")

            def metric_reader(event_files):
                self.assertEqual([path.name for path in event_files], ["events.out.tfevents.fake"])
                return [
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
                ]

            runs = LocalRunIndexer(root, metric_reader=metric_reader).discover_runs()

        self.assertEqual(runs[0].metric_summaries[0].tag, "Train/mean_reward")
        self.assertEqual(runs[0].metric_summaries[0].last_value, 2.0)

    def test_discover_runs_uses_metric_series_reader_for_event_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "group" / "run-a"
            run_dir.mkdir(parents=True)
            event_file = run_dir / "events.out.tfevents.fake"
            event_file.write_text("event", encoding="utf-8")

            def metric_series_reader(event_files):
                self.assertEqual([path.name for path in event_files], ["events.out.tfevents.fake"])
                return [
                    MetricSeries(
                        tag="Train/mean_reward",
                        points=[{"step": 0, "value": 1.0}, {"step": 10, "value": 2.0}],
                        original_count=2,
                    )
                ]

            runs = LocalRunIndexer(root, metric_series_reader=metric_series_reader).discover_runs()

        self.assertEqual(runs[0].metric_series[0].tag, "Train/mean_reward")
        self.assertEqual(runs[0].metric_series[0].points[-1]["value"], 2.0)

    def test_discover_runs_infers_parent_from_resume_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "group" / "baseline"
            child = root / "group" / "child"
            (parent / "params").mkdir(parents=True)
            (child / "params").mkdir(parents=True)
            (parent / "params" / "agent.yaml").write_text("agent:\n  seed: 1\n", encoding="utf-8")
            (child / "params" / "agent.yaml").write_text(
                "\n".join(
                    [
                        "agent:",
                        "  runner:",
                        "    resume: true",
                        "    load_run: baseline",
                        "    load_checkpoint: model_100.pt",
                    ]
                ),
                encoding="utf-8",
            )

            runs = {run.run_id: run for run in LocalRunIndexer(root).discover_runs()}

        self.assertEqual(runs["group/child"].parent_run_id, "group/baseline")
        self.assertEqual(runs["group/child"].parent_checkpoint, "model_100.pt")

    def test_discover_runs_extracts_explicit_git_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "group" / "run-with-git"
            params_dir = run_dir / "params"
            params_dir.mkdir(parents=True)
            (params_dir / "git.yaml").write_text(
                "\n".join(
                    [
                        "git:",
                        "  commit: abc123",
                        "  branch: rl-dashboard",
                        "  dirty: true",
                        "  diff: changed rewards",
                    ]
                ),
                encoding="utf-8",
            )

            run = LocalRunIndexer(root).discover_runs()[0]

        self.assertEqual(run.git_metadata["commit"], "abc123")
        self.assertEqual(run.git_metadata["branch"], "rl-dashboard")
        self.assertEqual(run.git_metadata["dirty"], True)
        self.assertEqual(run.git_metadata["diff"], "changed rewards")


if __name__ == "__main__":
    unittest.main()
