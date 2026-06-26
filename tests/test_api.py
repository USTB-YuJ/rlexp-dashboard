import tempfile
import unittest
from pathlib import Path

from rl_exp_dashboard.api import (
    artifact_file_path,
    compare_runs_payload,
    index_project_payload,
    lineage_overview_payload,
    metric_series_payload,
    metric_summaries_payload,
    projects_payload,
    remote_sources_payload,
    run_report_markdown,
    run_detail_payload,
    runs_payload,
    save_checkpoint_review_payload,
    save_lineage_edge_payload,
    save_project_payload,
    save_remote_source_payload,
    save_run_observation_payload,
    sync_remote_source_payload,
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
                tags=["stairs"],
                score=0.72,
                recommended=True,
            )

            payload = runs_payload(store, "project")

        by_run_id = {run["run_id"]: run for run in payload["runs"]}
        child = by_run_id["group/child"]
        parent = by_run_id["group/parent"]
        self.assertEqual(child["task_name"], "Unitree-G1-Depth-Parkour")
        self.assertEqual(child["algorithm_name"], "rsl_rl_ppo")
        self.assertEqual(child["review_verdict"], "mixed")
        self.assertEqual(child["recommended_checkpoint"], "model_20.pt")
        self.assertEqual(child["recommended_review_checkpoint"], "model_20.pt")
        self.assertEqual(child["best_review_checkpoint"], "model_20.pt")
        self.assertEqual(child["best_review_score"], 0.72)
        self.assertEqual(child["best_review_status"], "mixed")
        self.assertEqual(child["review_tags"], ["reviewed"])
        self.assertEqual(child["checkpoint_review_tags"], ["stairs"])
        self.assertEqual(child["all_tags"], ["reviewed", "stairs"])
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
        self.assertEqual(child["start_time"], 200.0)
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
        self.assertEqual(payload["checkpoints"][0]["metric_values"][0]["tag"], "Train/mean_reward")
        self.assertEqual(payload["checkpoints"][0]["metric_values"][0]["value"], 2.5)
        self.assertEqual(payload["checkpoints"][0]["metric_values"][0]["step"], 20)
        self.assertEqual(payload["checkpoints"][0]["metric_values"][0]["step_delta"], 0)
        self.assertEqual(payload["metrics"][0]["tag"], "Train/mean_reward")
        self.assertEqual(payload["parent_lineage"][0]["parent_run_id"], "group/parent")
        self.assertEqual(payload["child_lineage"], [])
        self.assertEqual(payload["observation"]["verdict"], "good")
        self.assertEqual(payload["checkpoint_reviews"][0]["checkpoint"], "model_20.pt")
        self.assertEqual([artifact["kind"] for artifact in payload["artifacts"]], ["artifact", "video"])
        self.assertEqual([Path(artifact["path"]).name for artifact in payload["artifacts"]], ["policy.onnx", "model_20.mp4"])
        self.assertEqual([item["relative_path"] for item in payload["config_files"]], ["params/agent.yaml", "params/env.yaml"])
        self.assertEqual([item["relative_path"] for item in payload["event_files"]], ["events.out.tfevents.fake"])

    def test_run_detail_payload_includes_reward_and_termination_config_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DashboardStore(tmp_path / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", tmp_path / "cache")
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/run",
                    name="run",
                    group="group",
                    path=tmp_path / "run",
                    modified_time=1.0,
                    params={
                        "env": {
                            "rewards": {
                                "track_lin_vel_xy_exp": {
                                    "weight": 1.5,
                                    "params": {"std": 0.5},
                                },
                                "joint_acc_l2": {"weight": -1e-7},
                            },
                            "terminations": {
                                "body_height": {
                                    "time_out": False,
                                    "params": {"min_height": 0.4},
                                },
                                "time_out": {"time_out": True},
                            },
                        }
                    },
                ),
            )

            payload = run_detail_payload(store, "group/run")

        rewards = {item["name"]: item for item in payload["config_summary"]["rewards"]}
        terminations = {item["name"]: item for item in payload["config_summary"]["terminations"]}
        self.assertEqual(rewards["track_lin_vel_xy_exp"]["weight"], 1.5)
        self.assertEqual(rewards["track_lin_vel_xy_exp"]["params"], {"std": 0.5})
        self.assertEqual(rewards["joint_acc_l2"]["weight"], -1e-7)
        self.assertEqual(rewards["joint_acc_l2"]["path"], "env.rewards.joint_acc_l2")
        self.assertEqual(terminations["body_height"]["time_out"], False)
        self.assertEqual(terminations["body_height"]["params"], {"min_height": 0.4})
        self.assertEqual(terminations["time_out"]["time_out"], True)

    def test_run_detail_checkpoint_metric_snapshot_uses_project_preferred_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DashboardStore(tmp_path / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", tmp_path / "cache", preferred_metrics=["Episode/length"])
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/run",
                    name="run",
                    group="group",
                    path=tmp_path / "run",
                    modified_time=1.0,
                    checkpoints=[
                        CheckpointRecord(
                            path=tmp_path / "run" / "model_20.pt",
                            iteration=20,
                            size_bytes=128,
                            modified_time=1.0,
                            is_latest=True,
                        )
                    ],
                    metric_series=[
                        MetricSeries(
                            tag="Train/mean_reward",
                            points=[{"step": 20, "value": 1.0}],
                            original_count=1,
                        ),
                        MetricSeries(
                            tag="Episode/length",
                            points=[{"step": 18, "value": 900.0}],
                            original_count=1,
                        ),
                    ],
                ),
            )

            payload = run_detail_payload(store, "group/run")

        metric_values = payload["checkpoints"][0]["metric_values"]
        self.assertEqual([metric["tag"] for metric in metric_values], ["Episode/length"])
        self.assertEqual(metric_values[0]["value"], 900.0)
        self.assertEqual(metric_values[0]["step"], 18)
        self.assertEqual(metric_values[0]["step_delta"], 2.0)

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

    def test_run_detail_payload_includes_child_branch_comparison_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = self._store_with_parent_and_child(tmp_path)
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/child-b",
                    name="child-b",
                    group="group",
                    path=tmp_path / "child-b",
                    modified_time=4.0,
                    params={"agent": {"algorithm": {"entropy_coef": 0.02}}},
                    metric_summaries=[
                        MetricSummary(
                            tag="Train/mean_reward",
                            first_step=0,
                            last_step=10,
                            first_value=0.5,
                            last_value=0.5,
                            min_value=0.5,
                            max_value=0.5,
                            count=1,
                        )
                    ],
                ),
            )
            store.upsert_lineage(
                LineageEdge(
                    parent_run_id="group/parent",
                    child_run_id="group/child-b",
                    relationship="ablation",
                    intended_change="higher entropy",
                    result_summary="reward regressed",
                )
            )
            store.upsert_run_observation(
                run_id="group/child-b",
                verdict="bad",
                summary="Unstable gait.",
                tags=["falling"],
            )

            payload = run_detail_payload(store, "group/parent")

        branches = {entry["child_run_id"]: entry for entry in payload["child_branch_compare"]}
        self.assertEqual(set(branches), {"group/child", "group/child-b"})
        self.assertEqual(branches["group/child"]["relationship"], "finetune")
        self.assertEqual(branches["group/child"]["config_diff_count"], 1)
        self.assertEqual(branches["group/child"]["metric_deltas"][0]["delta_last_value"], 1.0)
        self.assertEqual(branches["group/child-b"]["relationship"], "ablation")
        self.assertEqual(branches["group/child-b"]["intended_change"], "higher entropy")
        self.assertEqual(branches["group/child-b"]["result_summary"], "reward regressed")
        self.assertEqual(branches["group/child-b"]["review_verdict"], "bad")
        self.assertEqual(branches["group/child-b"]["review_summary"], "Unstable gait.")
        self.assertEqual(branches["group/child-b"]["metric_deltas"][0]["delta_last_value"], -0.5)

    def test_run_report_markdown_summarizes_lineage_metrics_and_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))
            store.upsert_run_observation(
                run_id="group/child",
                verdict="mixed",
                summary="Good flat gait but still weak on stairs.",
                tags=["good-flat", "bad-stairs"],
                recommended_checkpoint="model_20.pt",
            )
            store.upsert_checkpoint_review(
                run_id="group/child",
                checkpoint="model_20.pt",
                status="mixed",
                notes="Stable forward walking, occasional stair stumble.",
                tags=["candidate"],
                score=0.72,
                recommended=True,
            )

            markdown = run_report_markdown(store, "group/child")

        self.assertIn("# Experiment Report: group/child", markdown)
        self.assertIn("- Task: Unitree-G1-Depth-Parkour", markdown)
        self.assertIn("- Algorithm: rsl_rl_ppo", markdown)
        self.assertIn("- Parent: group/parent", markdown)
        self.assertIn("- Parent checkpoint: model_10.pt", markdown)
        self.assertIn("- Intended change: lower entropy", markdown)
        self.assertIn("| Train/mean_reward | 2.0 | 10 |", markdown)
        self.assertIn("- Verdict: mixed", markdown)
        self.assertIn("- Tags: bad-stairs, good-flat", markdown)
        self.assertIn("| model_20.pt | mixed | 0.72 | yes | candidate |", markdown)
        self.assertIn("Stable forward walking, occasional stair stumble.", markdown)

    def test_run_report_markdown_summarizes_child_branch_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = self._store_with_parent_and_child(tmp_path)
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/child-b",
                    name="child-b",
                    group="group",
                    path=tmp_path / "child-b",
                    modified_time=4.0,
                    params={"agent": {"algorithm": {"entropy_coef": 0.02}}},
                    metric_summaries=[
                        MetricSummary(
                            tag="Train/mean_reward",
                            first_step=0,
                            last_step=10,
                            first_value=0.5,
                            last_value=0.5,
                            min_value=0.5,
                            max_value=0.5,
                            count=1,
                        )
                    ],
                ),
            )
            store.upsert_lineage(
                LineageEdge(
                    parent_run_id="group/parent",
                    child_run_id="group/child-b",
                    relationship="ablation",
                    intended_change="higher entropy",
                    result_summary="reward regressed",
                )
            )
            store.upsert_run_observation(
                run_id="group/child-b",
                verdict="bad",
                summary="Unstable gait.",
                tags=["falling"],
            )

            markdown = run_report_markdown(store, "group/parent")

        self.assertIn("## Child Branch Comparison", markdown)
        self.assertIn("| group/child | finetune | lower entropy | unreviewed | 1 | Train/mean_reward: 1.0 |", markdown)
        self.assertIn("| group/child-b | ablation | higher entropy | bad: Unstable gait. | 1 | Train/mean_reward: -0.5 |", markdown)
        self.assertIn("reward regressed", markdown)

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
                        "commands": {"base_velocity": {"heading_kp": 1.0}},
                        "terrain": {"generator": {"curriculum": False}},
                        "observations": {"actor": {"depth": True}},
                        "curriculum": {"terrain_levels": {"enabled": False}},
                        "amp": {"motion_file": "walk1.npz"},
                        "estimator": {"contact_loss_coef": 1.0},
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
                        "commands": {"base_velocity": {"heading_kp": 0.5}},
                        "terrain": {"generator": {"curriculum": True}},
                        "observations": {"actor": {"depth": False}},
                        "curriculum": {"terrain_levels": {"enabled": True}},
                        "amp": {"motion_file": "walk_best.npz"},
                        "estimator": {"contact_loss_coef": 0.5},
                    },
                ),
            )

            payload = compare_runs_payload(store, "group/baseline", "group/target")

        groups = {group["key"]: group for group in payload["config_diff_groups"]}
        self.assertEqual(groups["rewards"]["title"], "Reward Diffs")
        self.assertEqual(groups["commands"]["title"], "Command Diffs")
        self.assertEqual(groups["terrain"]["title"], "Terrain Diffs")
        self.assertEqual(groups["algorithm"]["title"], "Algorithm Diffs")
        self.assertEqual(groups["amp"]["title"], "AMP Diffs")
        self.assertEqual(groups["estimator"]["title"], "Estimator Diffs")
        self.assertEqual(groups["observation_network"]["title"], "Observation / Network Diffs")
        self.assertEqual(groups["curriculum_termination"]["title"], "Curriculum / Termination Diffs")
        self.assertEqual(groups["rewards"]["diffs"][0]["path"], "rewards.action_rate_l2.weight")
        self.assertEqual(groups["commands"]["diffs"][0]["path"], "commands.base_velocity.heading_kp")
        self.assertEqual(groups["terrain"]["diffs"][0]["path"], "terrain.generator.curriculum")
        self.assertEqual(groups["algorithm"]["diffs"][0]["path"], "agent.algorithm.entropy_coef")
        self.assertEqual(groups["amp"]["diffs"][0]["path"], "amp.motion_file")
        self.assertEqual(groups["estimator"]["diffs"][0]["path"], "estimator.contact_loss_coef")
        self.assertEqual(groups["observation_network"]["diffs"][0]["path"], "observations.actor.depth")
        self.assertEqual(groups["curriculum_termination"]["diffs"][0]["path"], "curriculum.terrain_levels.enabled")

    def test_compare_runs_payload_returns_config_diff_and_metric_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store_with_parent_and_child(Path(tmp))
            store.upsert_run_observation(
                run_id="group/parent",
                verdict="bad",
                summary="Baseline falls on stairs.",
                tags=["falling"],
            )
            store.upsert_run_observation(
                run_id="group/child",
                verdict="mixed",
                summary="Better rough walking, still weak on stairs.",
                tags=["good-rough", "bad-stairs"],
                recommended_checkpoint="model_20.pt",
            )
            store.upsert_checkpoint_review(
                run_id="group/child",
                checkpoint="model_20.pt",
                status="mixed",
                notes="Candidate for more stair training.",
                tags=["candidate"],
                score=0.72,
                recommended=True,
            )

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
        self.assertEqual(payload["metric_series_compare"][0]["tag"], "Train/mean_reward")
        self.assertEqual(payload["metric_series_compare"][0]["before"]["points"][0]["value"], 1.0)
        self.assertEqual(payload["metric_series_compare"][0]["after"]["points"][-1]["value"], 2.0)
        self.assertEqual(payload["metric_series_compare"][0]["before"]["run_id"], "group/parent")
        self.assertEqual(payload["metric_series_compare"][0]["after"]["run_id"], "group/child")
        self.assertEqual(payload["review_compare"]["before"]["verdict"], "bad")
        self.assertEqual(payload["review_compare"]["before"]["summary"], "Baseline falls on stairs.")
        self.assertEqual(payload["review_compare"]["before"]["tags"], ["falling"])
        self.assertEqual(payload["review_compare"]["before"]["checkpoint_review_count"], 0)
        self.assertEqual(payload["review_compare"]["after"]["verdict"], "mixed")
        self.assertEqual(payload["review_compare"]["after"]["summary"], "Better rough walking, still weak on stairs.")
        self.assertEqual(payload["review_compare"]["after"]["tags"], ["bad-stairs", "good-rough"])
        self.assertEqual(payload["review_compare"]["after"]["recommended_checkpoint"], "model_20.pt")
        self.assertEqual(payload["review_compare"]["after"]["checkpoint_review_count"], 1)
        self.assertEqual(payload["review_compare"]["after"]["recommended_review_checkpoint"], "model_20.pt")
        self.assertEqual(payload["review_compare"]["after"]["best_review_score"], 0.72)

    def test_compare_runs_payload_includes_video_artifact_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = DashboardStore(tmp_path / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", tmp_path / "cache")
            before_video = tmp_path / "before" / "videos" / "play" / "model_10.mp4"
            before_artifact = tmp_path / "before" / "exports" / "policy.onnx"
            after_video = tmp_path / "after" / "videos" / "play" / "model_20.mp4"
            before_video.parent.mkdir(parents=True)
            before_artifact.parent.mkdir(parents=True)
            after_video.parent.mkdir(parents=True)
            before_video.write_bytes(b"before-video")
            before_artifact.write_bytes(b"before-onnx")
            after_video.write_bytes(b"after-video")
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/before",
                    name="before",
                    group="group",
                    path=tmp_path / "before",
                    modified_time=1.0,
                    videos=[before_video],
                    artifacts=[before_artifact],
                ),
            )
            store.upsert_run(
                "project",
                RunRecord(
                    run_id="group/after",
                    name="after",
                    group="group",
                    path=tmp_path / "after",
                    modified_time=2.0,
                    videos=[after_video],
                ),
            )

            payload = compare_runs_payload(store, "group/before", "group/after")

        self.assertEqual(payload["artifact_compare"]["before"]["run_id"], "group/before")
        self.assertEqual(payload["artifact_compare"]["after"]["run_id"], "group/after")
        self.assertEqual(
            [Path(item["path"]).name for item in payload["artifact_compare"]["before"]["videos"]],
            ["model_10.mp4"],
        )
        self.assertEqual(
            [Path(item["path"]).name for item in payload["artifact_compare"]["after"]["videos"]],
            ["model_20.mp4"],
        )
        self.assertEqual(
            [Path(item["path"]).name for item in payload["artifact_compare"]["before"]["artifacts"]],
            ["policy.onnx"],
        )

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
                    "confirmation_state": "confirmed",
                    "confidence_source": "manual",
                    "result_summary": "Better target success but rough gait.",
                },
            )
            overview = lineage_overview_payload(store, "project")

        self.assertEqual(saved["edge"]["relationship"], "manual-link")
        self.assertEqual(saved["edge"]["confirmed"], True)
        self.assertEqual(saved["edge"]["confirmation_state"], "confirmed")
        self.assertEqual(saved["edge"]["confidence_source"], "manual")
        self.assertEqual(saved["edge"]["result_summary"], "Better target success but rough gait.")
        self.assertEqual(overview["edges"][0]["note"], "Linked by user.")
        self.assertEqual(overview["edges"][0]["result_summary"], "Better target success but rough gait.")

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

    def test_save_remote_source_payload_persists_dashboard_form_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DashboardStore(Path(tmp) / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", Path(tmp) / "cache")

            saved = save_remote_source_payload(
                store,
                {
                    "name": "x-server",
                    "project": "project",
                    "host": "1858d9aa66b16579.natapp.cc",
                    "user": "eai",
                    "port": "12188",
                    "remote_log_root": "/home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl",
                    "method": "scp",
                    "include_patterns": "params/***, events.out.tfevents.*",
                    "exclude_patterns": ["videos/***", "wandb/***"],
                },
            )
            payload = remote_sources_payload(store, "project")

        source = payload["sources"][0]
        self.assertEqual(saved["source"]["name"], "x-server")
        self.assertEqual(source["host"], "1858d9aa66b16579.natapp.cc")
        self.assertEqual(source["user"], "eai")
        self.assertEqual(source["port"], 12188)
        self.assertEqual(source["remote_log_root"], "/home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl")
        self.assertEqual(source["method"], "scp")
        self.assertEqual(source["include_patterns"], ["params/***", "events.out.tfevents.*"])
        self.assertEqual(source["exclude_patterns"], ["videos/***", "wandb/***"])

    def test_sync_remote_source_payload_records_dry_run_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DashboardStore(Path(tmp) / "dashboard.sqlite3")
            store.initialize()
            cache_root = Path(tmp) / "cache"
            store.upsert_project("project", cache_root)
            store.upsert_remote_source(
                RemoteSource(
                    name="x-server",
                    host="example.com",
                    user="eai",
                    port=12188,
                    remote_log_root="/logs/rsl_rl",
                    project="project",
                    method="rsync",
                )
            )

            payload = sync_remote_source_payload(
                store,
                {
                    "project": "project",
                    "source_name": "x-server",
                    "dry_run": True,
                    "include_checkpoints": True,
                },
            )
            status = store.latest_sync_status("x-server", "project")

        self.assertEqual(payload["status"], "dry-run")
        self.assertEqual(payload["source_name"], "x-server")
        self.assertEqual(payload["project"], "project")
        self.assertIn("rsync", payload["command"])
        self.assertIn("--dry-run", payload["command"])
        self.assertIn("--include=model_*.pt", payload["command"])
        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["local_path"], str(cache_root / "x-server" / "project" / "logs" / "rsl_rl"))

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

    def test_index_project_payload_discovers_runs_and_persists_lineage(self):
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
            (child / "model_20.pt").write_bytes(b"checkpoint")
            store = DashboardStore(tmp_path / "dashboard.sqlite3")
            store.initialize()
            store.upsert_project("project", log_root)

            payload = index_project_payload(store, {"project": "project"})

            runs = runs_payload(store, "project")
            lineage = lineage_overview_payload(store, "project")

        self.assertEqual(payload["project"], "project")
        self.assertEqual(payload["log_root"], str(log_root))
        self.assertEqual(payload["indexed_run_count"], 2)
        self.assertEqual({run["run_id"] for run in runs["runs"]}, {"group/baseline", "group/child"})
        self.assertEqual(lineage["edges"][0]["parent_run_id"], "group/baseline")
        self.assertEqual(lineage["edges"][0]["parent_checkpoint"], "model_100.pt")

    def _store_with_parent_and_child(self, tmp_path: Path) -> DashboardStore:
        store = DashboardStore(tmp_path / "dashboard.sqlite3")
        store.initialize()
        store.upsert_project("project", tmp_path / "cache")
        child_video = tmp_path / "child" / "videos" / "play" / "model_20.mp4"
        child_artifact = tmp_path / "child" / "exports" / "policy.onnx"
        child_env_config = tmp_path / "child" / "params" / "env.yaml"
        child_agent_config = tmp_path / "child" / "params" / "agent.yaml"
        child_event_file = tmp_path / "child" / "events.out.tfevents.fake"
        child_video.parent.mkdir(parents=True)
        child_artifact.parent.mkdir(parents=True)
        child_env_config.parent.mkdir(parents=True)
        child_video.write_bytes(b"video")
        child_artifact.write_bytes(b"onnx")
        child_env_config.write_text("env:\n  rewards: {}\n", encoding="utf-8")
        child_agent_config.write_text("agent:\n  algorithm: {}\n", encoding="utf-8")
        child_event_file.write_text("event", encoding="utf-8")
        store.upsert_run(
            "project",
            RunRecord(
                run_id="group/parent",
                name="parent",
                group="group",
                path=tmp_path / "parent",
                modified_time=3.0,
                start_time=100.0,
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
                start_time=200.0,
                task_name="Unitree-G1-Depth-Parkour",
                algorithm_name="rsl_rl_ppo",
                params={"agent": {"algorithm": {"entropy_coef": 0.005}}},
                param_files=[child_env_config, child_agent_config],
                event_files=[child_event_file],
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
                        points=[
                            {"step": 0, "value": 1.0},
                            {"step": 20, "value": 2.5},
                            {"step": 30, "value": 2.0},
                        ],
                        original_count=3,
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
