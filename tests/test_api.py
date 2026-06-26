import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.api import (
    artifact_file_path,
    compare_runs_payload,
    lineage_overview_payload,
    metric_series_payload,
    metric_summaries_payload,
    projects_payload,
    remote_sources_payload,
    run_detail_payload,
    runs_payload,
    save_checkpoint_review_payload,
    save_lineage_edge_payload,
    save_project_payload,
    save_run_observation_payload,
    timeline_payload,
)
from rl_exp_dashboard.models import CheckpointRecord, LineageEdge, MetricSeries, MetricSummary, RunRecord
from rl_exp_dashboard.storage import DashboardStore
from rl_exp_dashboard.sync import RemoteSource


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

    def test_runs_payload_includes_table_summary_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))
            store.upsert_project(
                "project",
                Path(tmp) / "cache",
                preferred_metrics=["Train/mean_reward"],
            )
            store.upsert_run_observation(
                run_id="group/child",
                verdict="mixed",
                summary="Good flat, still weak stairs.",
                tags=["reviewed"],
                recommended_checkpoint="model_20.pt",
            )
            store.upsert_checkpoint_review(
                run_id="group/child",
                checkpoint="model_20.pt",
                status="mixed",
                notes="Visual review done.",
                tags=["reviewed"],
            )

            payload = runs_payload(store, "project")

        by_run_id = {run["run_id"]: run for run in payload["runs"]}
        child = by_run_id["group/child"]
        parent = by_run_id["group/parent"]
        self.assertEqual(child["review_verdict"], "mixed")
        self.assertEqual(child["recommended_checkpoint"], "model_20.pt")
        self.assertEqual(child["has_observation"], True)
        self.assertEqual(child["has_reviewed_checkpoint"], True)
        self.assertEqual(child["video_count"], 1)
        self.assertEqual(child["artifact_count"], 1)
        self.assertEqual(child["parent_count"], 1)
        self.assertEqual(child["child_count"], 0)
        self.assertEqual(parent["child_count"], 1)
        self.assertEqual(child["key_metrics"]["Train/mean_reward"]["last_value"], 2.0)
        self.assertEqual(child["key_metrics"]["Train/mean_reward"]["last_step"], 10)

    def test_timeline_payload_returns_chronological_experiment_story(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))
            store.upsert_project(
                "project",
                Path(tmp) / "cache",
                preferred_metrics=["Train/mean_reward"],
            )
            store.upsert_run_observation(
                run_id="group/child",
                verdict="bad",
                summary="Backward walking in play.",
                tags=["backward-walk"],
                recommended_checkpoint="model_20.pt",
            )

            payload = timeline_payload(store, "project")

        self.assertEqual([entry["run_id"] for entry in payload["entries"]], ["group/parent", "group/child"])
        child = payload["entries"][1]
        self.assertEqual(child["parent_run_id"], "group/parent")
        self.assertEqual(child["parent_checkpoint"], "model_10.pt")
        self.assertEqual(child["relationship"], "finetune")
        self.assertEqual(child["intended_change"], "lower entropy")
        self.assertEqual(child["review_verdict"], "bad")
        self.assertEqual(child["summary"], "Backward walking in play.")
        self.assertEqual(child["tags"], ["backward-walk"])
        self.assertEqual(child["key_metrics"]["Train/mean_reward"]["last_value"], 2.0)

    def test_metric_series_payload_returns_sampled_points(self):
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
                    metric_series=[
                        MetricSeries(
                            tag="Train/mean_reward",
                            points=[{"step": 0, "value": 1.0}, {"step": 10, "value": 2.0}],
                            original_count=2,
                        ),
                        MetricSeries(
                            tag="Episode/length",
                            points=[{"step": 0, "value": 100.0}],
                            original_count=1,
                        ),
                    ],
                ),
            )

            payload = metric_series_payload(store, "group/run", tag="Train/mean_reward")

        self.assertEqual(payload["run_id"], "group/run")
        self.assertEqual(len(payload["series"]), 1)
        self.assertEqual(payload["series"][0]["tag"], "Train/mean_reward")
        self.assertEqual(payload["series"][0]["points"][-1]["value"], 2.0)

    def test_run_detail_payload_combines_run_checkpoints_metrics_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))
            store.upsert_run_observation(
                run_id="group/child",
                verdict="good",
                summary="Good candidate.",
                tags=["candidate"],
                recommended_checkpoint="model_20.pt",
            )
            store.upsert_checkpoint_review(
                run_id="group/child",
                checkpoint="model_20.pt",
                status="good",
                notes="Stable in play.",
                tags=["forward-walk"],
                recommended=True,
            )
            payload = run_detail_payload(store, "group/child")

        self.assertEqual(payload["run"]["run_id"], "group/child")
        self.assertEqual(payload["checkpoints"][0]["iteration"], 20)
        self.assertEqual(payload["metrics"][0]["tag"], "Train/mean_reward")
        self.assertEqual(payload["parent_lineage"][0]["parent_run_id"], "group/parent")
        self.assertEqual(payload["child_lineage"], [])
        self.assertEqual(payload["observation"]["verdict"], "good")
        self.assertEqual(payload["checkpoint_reviews"][0]["checkpoint"], "model_20.pt")
        self.assertEqual([artifact["kind"] for artifact in payload["artifacts"]], ["artifact", "video"])
        self.assertEqual([Path(artifact["path"]).name for artifact in payload["artifacts"]], ["policy.onnx", "model_20.mp4"])

    def test_run_detail_payload_includes_parent_comparison_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))

            payload = run_detail_payload(store, "group/child")

        parent_compare = payload["parent_compare"]
        groups = {group["key"]: group for group in parent_compare["config_diff_groups"]}
        deltas = {delta["tag"]: delta for delta in parent_compare["metric_deltas"]}
        self.assertEqual(parent_compare["parent_run_id"], "group/parent")
        self.assertEqual(parent_compare["child_run_id"], "group/child")
        self.assertEqual(parent_compare["relationship"], "finetune")
        self.assertEqual(parent_compare["parent_checkpoint"], "model_10.pt")
        self.assertEqual(parent_compare["intended_change"], "lower entropy")
        self.assertEqual(groups["algorithm"]["diffs"][0]["path"], "agent.algorithm.entropy_coef")
        self.assertEqual(groups["algorithm"]["diffs"][0]["before"], 0.01)
        self.assertEqual(groups["algorithm"]["diffs"][0]["after"], 0.005)
        self.assertEqual(deltas["Train/mean_reward"]["delta_last_value"], 1.0)

    def test_artifact_file_path_only_allows_indexed_run_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))
            allowed = Path(tmp) / "child" / "videos" / "play" / "model_20.mp4"
            denied = Path(tmp) / "child" / "secret.txt"
            denied.write_text("secret", encoding="utf-8")

            resolved = artifact_file_path(store, "group/child", str(allowed))

            with self.assertRaises(PermissionError):
                artifact_file_path(store, "group/child", str(denied))

        self.assertEqual(resolved, allowed)

    def test_save_observation_and_checkpoint_review_payloads_persist_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))

            observation_payload = save_run_observation_payload(
                store,
                {
                    "run_id": "group/child",
                    "verdict": "mixed",
                    "summary": "Good flat, bad stairs.",
                    "tags": ["good-rough", "bad-stairs"],
                    "recommended_checkpoint": "model_20.pt",
                },
            )
            review_payload = save_checkpoint_review_payload(
                store,
                {
                    "run_id": "group/child",
                    "checkpoint": "model_20.pt",
                    "status": "mixed",
                    "notes": "Needs stairs work.",
                    "tags": ["bad-stairs"],
                    "video_path": "videos/model_20.mp4",
                    "score": 0.5,
                    "recommended": False,
                },
            )

            detail = run_detail_payload(store, "group/child")

        self.assertEqual(observation_payload["observation"]["verdict"], "mixed")
        self.assertEqual(review_payload["review"]["status"], "mixed")
        self.assertEqual(detail["observation"]["tags"], ["good-rough", "bad-stairs"])
        self.assertEqual(detail["checkpoint_reviews"][0]["video_path"], "videos/model_20.mp4")

    def test_compare_runs_payload_groups_config_diffs_by_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DashboardStore(Path(tmp) / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", Path(tmp) / "cache")
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/baseline",
                    name="baseline",
                    group="group",
                    path=Path(tmp) / "baseline",
                    modified_time=1.0,
                    params={
                        "rewards": {"action_rate_l2": {"weight": -0.0001}},
                        "agent": {"algorithm": {"entropy_coef": 0.01}},
                        "observations": {"actor": {"depth": True}},
                        "curriculum": {"terrain_levels": {"enabled": False}},
                    },
                ),
            )
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/target",
                    name="target",
                    group="group",
                    path=Path(tmp) / "target",
                    modified_time=2.0,
                    params={
                        "rewards": {"action_rate_l2": {"weight": -0.001}},
                        "agent": {"algorithm": {"entropy_coef": 0.005}},
                        "observations": {"actor": {"depth": False}},
                        "curriculum": {"terrain_levels": {"enabled": True}},
                    },
                ),
            )

            payload = compare_runs_payload(store, "group/baseline", "group/target")

        groups = {group["key"]: group for group in payload["config_diff_groups"]}
        self.assertEqual(groups["rewards"]["title"], "Reward Diffs")
        self.assertEqual(groups["algorithm"]["title"], "Algorithm Diffs")
        self.assertEqual(groups["observation_network"]["title"], "Observation / Network Diffs")
        self.assertEqual(groups["curriculum_termination"]["title"], "Curriculum / Termination Diffs")
        self.assertEqual(groups["rewards"]["diffs"][0]["path"], "rewards.action_rate_l2.weight")
        self.assertEqual(groups["algorithm"]["diffs"][0]["path"], "agent.algorithm.entropy_coef")
        self.assertEqual(groups["observation_network"]["diffs"][0]["path"], "observations.actor.depth")
        self.assertEqual(groups["curriculum_termination"]["diffs"][0]["path"], "curriculum.terrain_levels.enabled")

    def test_compare_runs_payload_returns_config_diff_and_metric_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))

            payload = compare_runs_payload(store, "group/parent", "group/child")
            diffs_by_path = {item["path"]: item for item in payload["config_diffs"]}
            deltas_by_tag = {item["tag"]: item for item in payload["metric_deltas"]}

        self.assertEqual(payload["before"]["run_id"], "group/parent")
        self.assertEqual(payload["after"]["run_id"], "group/child")
        self.assertEqual(diffs_by_path["agent.algorithm.entropy_coef"]["kind"], "changed")
        self.assertEqual(diffs_by_path["agent.algorithm.entropy_coef"]["before"], 0.01)
        self.assertEqual(diffs_by_path["agent.algorithm.entropy_coef"]["after"], 0.005)
        self.assertEqual(deltas_by_tag["Train/mean_reward"]["before_last_value"], 1.0)
        self.assertEqual(deltas_by_tag["Train/mean_reward"]["after_last_value"], 2.0)
        self.assertEqual(deltas_by_tag["Train/mean_reward"]["delta_last_value"], 1.0)

    def test_lineage_overview_payload_returns_nodes_and_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))

            payload = lineage_overview_payload(store, "project")

        self.assertEqual({node["run_id"] for node in payload["nodes"]}, {"group/parent", "group/child"})
        self.assertEqual(payload["edges"][0]["parent_run_id"], "group/parent")
        self.assertEqual(payload["edges"][0]["child_run_id"], "group/child")
        self.assertEqual(payload["edges"][0]["relationship"], "finetune")

    def test_save_lineage_edge_payload_persists_manual_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))

            saved = save_lineage_edge_payload(
                store,
                {
                    "parent_run_id": "group/parent",
                    "child_run_id": "group/child",
                    "relationship": "manual-link",
                    "parent_checkpoint": "model_10.pt",
                    "intended_change": "Corrected lineage after review.",
                    "note": "Linked by user.",
                },
            )
            overview = lineage_overview_payload(store, "project")

        self.assertEqual(saved["edge"]["relationship"], "manual-link")
        self.assertEqual(saved["edge"]["confirmed"], True)
        self.assertEqual(overview["edges"][0]["note"], "Linked by user.")

    def test_remote_sources_payload_includes_latest_sync_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DashboardStore(Path(tmp) / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", Path(tmp) / "cache")
            store.upsert_remote_source(
                RemoteSource(
                    name="x-server",
                    host="example.com",
                    user="eai",
                    port=12188,
                    remote_log_root="/logs/rsl_rl",
                    project="project",
                    include_patterns=("params/***",),
                    exclude_patterns=("videos/***",),
                )
            )
            store.record_sync_status(
                source_name="x-server",
                project="project",
                status="dry-run",
                command=["rsync", "--dry-run"],
                local_path=Path(tmp) / "cache",
            )

            payload = remote_sources_payload(store, "project")

        self.assertEqual(payload["sources"][0]["name"], "x-server")
        self.assertEqual(payload["sources"][0]["include_patterns"], ["params/***"])
        self.assertEqual(payload["sources"][0]["exclude_patterns"], ["videos/***"])
        self.assertEqual(payload["sources"][0]["latest_sync"]["status"], "dry-run")
        self.assertEqual(payload["sources"][0]["latest_sync"]["command"], ["rsync", "--dry-run"])

    def test_projects_payload_includes_project_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DashboardStore(Path(tmp) / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project(
                "project",
                Path(tmp) / "cache",
                parser_profile="rsl_rl_tensorboard",
                preferred_metrics=["Train/mean_reward"],
                log_patterns=["logs/rsl_rl/*/*"],
                tag_schema=["good-flat"],
            )

            payload = projects_payload(store)

        self.assertEqual(payload["projects"][0]["name"], "project")
        self.assertEqual(payload["projects"][0]["parser_profile"], "rsl_rl_tensorboard")
        self.assertEqual(payload["projects"][0]["preferred_metrics"], ["Train/mean_reward"])

    def test_save_project_payload_persists_dashboard_project_form_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DashboardStore(Path(tmp) / "dashboard.sqlite3")
            store.initialize()

            saved = save_project_payload(
                store,
                {
                    "name": "unitree_rl_mjlab",
                    "local_cache_root": str(Path(tmp) / "cache"),
                    "parser_profile": "rsl_rl_tensorboard",
                    "preferred_metrics": "Train/mean_reward, Episode/length",
                    "log_patterns": "logs/rsl_rl/*/*\nlogs/other/*/*",
                    "tag_schema": ["good-flat", "bad-stairs"],
                },
            )
            payload = projects_payload(store)

        project = payload["projects"][0]
        self.assertEqual(saved["project"]["name"], "unitree_rl_mjlab")
        self.assertEqual(project["local_cache_root"], str(Path(tmp) / "cache"))
        self.assertEqual(project["parser_profile"], "rsl_rl_tensorboard")
        self.assertEqual(project["preferred_metrics"], ["Train/mean_reward", "Episode/length"])
        self.assertEqual(project["log_patterns"], ["logs/rsl_rl/*/*", "logs/other/*/*"])
        self.assertEqual(project["tag_schema"], ["good-flat", "bad-stairs"])

    def _store_with_parent_and_child(self, tmp_path: Path) -> DashboardStore:
        store = DashboardStore(tmp_path / "dashboard.sqlite3")
        store.initialize()
        store.upsert_project("project", tmp_path / "cache")
        child_video = tmp_path / "child" / "videos" / "play" / "model_20.mp4"
        child_artifact = tmp_path / "child" / "exports" / "policy.onnx"
        child_video.parent.mkdir(parents=True)
        child_artifact.parent.mkdir(parents=True)
        child_video.write_bytes(b"video")
        child_artifact.write_bytes(b"onnx")
        store.upsert_run(
            "project",
            RunRecord(
                run_id="group/parent",
                name="parent",
                group="group",
                path=tmp_path / "parent",
                modified_time=1.0,
                params={"agent": {"algorithm": {"entropy_coef": 0.01}}},
                metric_summaries=[
                    MetricSummary(
                        tag="Train/mean_reward",
                        first_step=0,
                        last_step=10,
                        first_value=1.0,
                        last_value=1.0,
                        min_value=1.0,
                        max_value=1.0,
                        count=1,
                    )
                ],
                metric_series=[
                    MetricSeries(
                        tag="Train/mean_reward",
                        points=[{"step": 0, "value": 1.0}],
                        original_count=1,
                    )
                ],
            ),
        )
        store.upsert_run(
            "project",
            RunRecord(
                run_id="group/child",
                name="child",
                group="group",
                path=tmp_path / "child",
                modified_time=2.0,
                params={"agent": {"algorithm": {"entropy_coef": 0.005}}},
                checkpoints=[
                    CheckpointRecord(
                        path=tmp_path / "child" / "model_20.pt",
                        iteration=20,
                        size_bytes=128,
                        modified_time=2.0,
                        is_latest=True,
                    )
                ],
                metric_summaries=[
                    MetricSummary(
                        tag="Train/mean_reward",
                        first_step=0,
                        last_step=10,
                        first_value=2.0,
                        last_value=2.0,
                        min_value=2.0,
                        max_value=2.0,
                        count=1,
                    )
                ],
                metric_series=[
                    MetricSeries(
                        tag="Train/mean_reward",
                        points=[{"step": 0, "value": 2.0}],
                        original_count=1,
                    )
                ],
                videos=[child_video],
                artifacts=[child_artifact],
            ),
        )
        store.upsert_lineage(
            LineageEdge(
                parent_run_id="group/parent",
                child_run_id="group/child",
                relationship="finetune",
                parent_checkpoint="model_10.pt",
                intended_change="lower entropy",
            )
        )
        return store


if __name__ == "__main__":
    unittest.main()
