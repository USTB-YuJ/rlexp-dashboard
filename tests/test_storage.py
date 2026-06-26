import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.models import CheckpointRecord, LineageEdge, MetricSeries, MetricSummary, RunRecord
from rl_exp_dashboard.storage import DashboardStore
from rl_exp_dashboard.sync import RemoteSource


class DashboardStoreTests(unittest.TestCase):
    def test_store_persists_runs_checkpoints_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "dashboard.sqlite3"
            run_path = Path(tmp) / "logs" / "group" / "child"
            run_path.mkdir(parents=True)
            checkpoint_path = run_path / "model_100.pt"
            video_path = run_path / "videos" / "play" / "rl-video-step-0.mp4"
            artifact_path = run_path / "exports" / "policy.onnx"
            checkpoint_path.write_bytes(b"checkpoint")
            video_path.parent.mkdir(parents=True)
            artifact_path.parent.mkdir(parents=True)
            video_path.write_bytes(b"video")
            artifact_path.write_bytes(b"onnx")

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
                    git_metadata={
                        "commit": "abc123",
                        "branch": "rl-dashboard",
                        "dirty": True,
                    },
                    checkpoints=[
                        CheckpointRecord(
                            path=checkpoint_path,
                            iteration=100,
                            size_bytes=10,
                            modified_time=3.0,
                            is_latest=True,
                        )
                    ],
                    metric_summaries=[
                        MetricSummary(
                            tag="Train/mean_reward",
                            first_step=0,
                            last_step=100,
                            first_value=1.0,
                            last_value=5.0,
                            min_value=1.0,
                            max_value=5.0,
                            count=2,
                            window_means={"mean100": 3.0},
                            slope_last_points=0.04,
                        )
                    ],
                    metric_series=[
                        MetricSeries(
                            tag="Train/mean_reward",
                            points=[{"step": 0, "value": 1.0}, {"step": 100, "value": 5.0}],
                            original_count=2,
                        )
                    ],
                    videos=[video_path],
                    artifacts=[artifact_path],
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
                    confirmation_state="suggested",
                    confidence_source="name-pattern",
                    result_summary="Needs human confirmation.",
                )
            )

            runs = store.list_runs("unitree_rl_mjlab")
            child = next(run for run in runs if run["run_id"] == "group/child")
            checkpoints = store.list_checkpoints("group/child")
            metrics = store.list_metric_summaries("group/child")
            series = store.list_metric_series("group/child")
            artifacts = store.list_run_artifacts("group/child")
            lineage = store.list_lineage("group/child")

        self.assertEqual(child["latest_checkpoint"], "model_100.pt")
        self.assertEqual(child["params"]["agent"]["algorithm"]["entropy_coef"], 0.005)
        self.assertEqual(child["git"]["commit"], "abc123")
        self.assertEqual(child["git"]["branch"], "rl-dashboard")
        self.assertEqual(child["git"]["dirty"], True)
        self.assertEqual(checkpoints[0]["iteration"], 100)
        self.assertEqual(checkpoints[0]["is_latest"], True)
        self.assertEqual(metrics[0]["tag"], "Train/mean_reward")
        self.assertEqual(metrics[0]["last_step"], 100)
        self.assertEqual(metrics[0]["last_value"], 5.0)
        self.assertEqual(metrics[0]["window_means"]["mean100"], 3.0)
        self.assertEqual(metrics[0]["slope_last_points"], 0.04)
        self.assertEqual(series[0]["tag"], "Train/mean_reward")
        self.assertEqual(series[0]["points"], [{"step": 0, "value": 1.0}, {"step": 100, "value": 5.0}])
        self.assertEqual(series[0]["original_count"], 2)
        self.assertEqual([artifact["kind"] for artifact in artifacts], ["artifact", "video"])
        self.assertEqual([Path(artifact["path"]).name for artifact in artifacts], ["policy.onnx", "rl-video-step-0.mp4"])
        self.assertEqual(artifacts[1]["suffix"], ".mp4")
        self.assertEqual(artifacts[1]["size_bytes"], 5)
        self.assertEqual(lineage[0]["parent_run_id"], "group/parent")
        self.assertEqual(lineage[0]["relationship"], "finetune")
        self.assertEqual(lineage[0]["intended_change"], "lower entropy")
        self.assertEqual(lineage[0]["confirmation_state"], "suggested")
        self.assertEqual(lineage[0]["confidence_source"], "name-pattern")
        self.assertEqual(lineage[0]["result_summary"], "Needs human confirmation.")
        self.assertEqual(lineage[0]["confirmed"], False)
        self.assertGreaterEqual(lineage[0]["updated_at"], 0.0)

    def test_store_lists_lineage_edges_scoped_by_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "dashboard.sqlite3"
            store = DashboardStore(db_path)
            store.initialize()
            store.upsert_project("project-a", Path(tmp) / "cache-a")
            store.upsert_project("project-b", Path(tmp) / "cache-b")
            for project, run_id in [
                ("project-a", "a/parent"),
                ("project-a", "a/child"),
                ("project-b", "b/parent"),
                ("project-b", "b/child"),
            ]:
                store.upsert_run(
                    project,
                    RunRecord(
                        run_id=run_id,
                        name=run_id.split("/")[-1],
                        group=run_id.split("/")[0],
                        path=Path(tmp) / run_id,
                        modified_time=1.0,
                    ),
                )
            store.upsert_lineage(
                LineageEdge(
                    parent_run_id="a/parent",
                    child_run_id="a/child",
                    relationship="resume",
                    intended_change="project a change",
                )
            )
            store.upsert_lineage(
                LineageEdge(
                    parent_run_id="b/parent",
                    child_run_id="b/child",
                    relationship="resume",
                    intended_change="project b change",
                )
            )

            project_edges = store.list_lineage_edges("project-a")
            all_edges = store.list_lineage_edges()

        self.assertEqual([edge["child_run_id"] for edge in project_edges], ["a/child"])
        self.assertEqual([edge["parent_run_id"] for edge in all_edges], ["a/parent", "b/parent"])

    def test_store_persists_remote_sources_and_sync_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "dashboard.sqlite3"
            store = DashboardStore(db_path)
            store.initialize()
            store.upsert_project("project", Path(tmp) / "cache")
            store.upsert_remote_source(
                RemoteSource(
                    name="x-server",
                    host="example.com",
                    user="robot",
                    port=2222,
                    remote_log_root="/logs/rsl_rl",
                    project="project",
                    method="rsync",
                    include_patterns=("params/***", "events.out.tfevents*"),
                    exclude_patterns=("videos/***", "tmp/***"),
                )
            )
            store.record_sync_status(
                source_name="x-server",
                project="project",
                status="dry-run",
                command=["rsync", "--dry-run"],
                local_path=Path(tmp) / "cache" / "x-server",
                message="preview only",
            )

            sources = store.list_remote_sources("project")
            status = store.latest_sync_status("x-server", "project")

        self.assertEqual(sources[0]["name"], "x-server")
        self.assertEqual(sources[0]["port"], 2222)
        self.assertEqual(sources[0]["remote_log_root"], "/logs/rsl_rl")
        self.assertEqual(sources[0]["include_patterns"], ["params/***", "events.out.tfevents*"])
        self.assertEqual(sources[0]["exclude_patterns"], ["videos/***", "tmp/***"])
        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["command"], ["rsync", "--dry-run"])
        self.assertEqual(status["message"], "preview only")

    def test_store_persists_project_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "dashboard.sqlite3"
            store = DashboardStore(db_path)
            store.initialize()

            store.upsert_project(
                "unitree_rl_mjlab",
                Path(tmp) / "cache",
                parser_profile="rsl_rl_tensorboard",
                preferred_metrics=["Train/mean_reward", "Episode_Termination/fell_over"],
                log_patterns=["logs/rsl_rl/*/*"],
                tag_schema=["good-flat", "bad-stairs"],
            )
            projects = store.list_projects()
            project = store.get_project("unitree_rl_mjlab")

        self.assertEqual(projects[0]["name"], "unitree_rl_mjlab")
        self.assertEqual(project["parser_profile"], "rsl_rl_tensorboard")
        self.assertEqual(project["preferred_metrics"], ["Train/mean_reward", "Episode_Termination/fell_over"])
        self.assertEqual(project["log_patterns"], ["logs/rsl_rl/*/*"])
        self.assertEqual(project["tag_schema"], ["good-flat", "bad-stairs"])

    def test_store_persists_run_observations_and_checkpoint_reviews(self):
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
                ),
            )

            store.upsert_run_observation(
                run_id="group/run",
                verdict="mixed",
                summary="Walks forward but sits on stairs.",
                tags=["forward-walk", "sitting"],
                recommended_checkpoint="model_100.pt",
            )
            store.upsert_checkpoint_review(
                run_id="group/run",
                checkpoint="model_100.pt",
                status="mixed",
                notes="Good on flat, bad on stairs.",
                tags=["good-rough", "bad-stairs"],
                video_path="videos/play/model_100.mp4",
                score=0.65,
                recommended=True,
            )

            observation = store.get_run_observation("group/run")
            reviews = store.list_checkpoint_reviews("group/run")

        self.assertEqual(observation["verdict"], "mixed")
        self.assertEqual(observation["summary"], "Walks forward but sits on stairs.")
        self.assertEqual(observation["tags"], ["forward-walk", "sitting"])
        self.assertEqual(observation["recommended_checkpoint"], "model_100.pt")
        self.assertEqual(reviews[0]["checkpoint"], "model_100.pt")
        self.assertEqual(reviews[0]["status"], "mixed")
        self.assertEqual(reviews[0]["tags"], ["good-rough", "bad-stairs"])
        self.assertEqual(reviews[0]["video_path"], "videos/play/model_100.mp4")
        self.assertEqual(reviews[0]["score"], 0.65)
        self.assertEqual(reviews[0]["recommended"], True)


if __name__ == "__main__":
    unittest.main()
