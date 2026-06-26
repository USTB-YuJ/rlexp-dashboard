import unittest
from pathlib import Path

from rl_exp_dashboard.sync import RemoteSource, build_sync_plan, execute_sync_plan


class RemoteSyncPlanTests(unittest.TestCase):
    def test_build_rsync_plan_uses_local_cache_and_default_filters(self):
        source = RemoteSource(
            name="x-server",
            host="1858d9aa66b16579.natapp.cc",
            user="eai",
            port=12188,
            remote_log_root="/home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl",
            project="unitree_rl_mjlab",
            method="rsync",
        )

        plan = build_sync_plan(source, cache_root=Path("/tmp/cache"), dry_run=True)

        self.assertEqual(plan.local_path, Path("/tmp/cache/x-server/unitree_rl_mjlab/logs/rsl_rl"))
        self.assertEqual(plan.method, "rsync")
        self.assertIn("--dry-run", plan.command)
        self.assertIn("-e", plan.command)
        self.assertIn("ssh -p 12188", plan.command)
        self.assertIn("--include=params/***", plan.command)
        self.assertIn("--include=events.out.tfevents*", plan.command)
        self.assertIn("--include=model_*.pt", plan.command)
        self.assertIn("--exclude=videos/***", plan.command)
        self.assertEqual(
            plan.command[-2:],
            [
                "eai@1858d9aa66b16579.natapp.cc:/home/eai/workspace/unitree_rl_mjlab/logs/rsl_rl/",
                "/tmp/cache/x-server/unitree_rl_mjlab/logs/rsl_rl/",
            ],
        )

    def test_build_scp_plan_can_include_videos_on_demand(self):
        source = RemoteSource(
            name="lab",
            host="example.com",
            user="robot",
            port=2222,
            remote_log_root="/logs/rsl_rl",
            project="project",
            method="scp",
        )

        plan = build_sync_plan(source, cache_root=Path("/cache"), dry_run=False, include_videos=True)

        self.assertEqual(plan.method, "scp")
        self.assertFalse(plan.dry_run)
        self.assertIn("-P", plan.command)
        self.assertIn("2222", plan.command)
        self.assertIn("robot@example.com:/logs/rsl_rl", plan.command)
        self.assertEqual(plan.include_videos, True)

    def test_execute_sync_plan_creates_local_path_and_invokes_runner(self):
        source = RemoteSource(
            name="lab",
            host="example.com",
            user="robot",
            port=2222,
            remote_log_root="/logs/rsl_rl",
            project="project",
            method="rsync",
        )
        plan = build_sync_plan(source, cache_root=Path("/tmp/cache"), dry_run=False)
        calls = []

        class FakeCompletedProcess:
            returncode = 0
            stdout = "copied"
            stderr = ""

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return FakeCompletedProcess()

        result = execute_sync_plan(plan, runner=runner)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.return_code, 0)
        self.assertEqual(result.stdout, "copied")
        self.assertEqual(calls[0][0], plan.command)
        self.assertEqual(calls[0][1]["capture_output"], True)
        self.assertEqual(calls[0][1]["text"], True)

    def test_execute_sync_plan_reports_failed_return_code(self):
        source = RemoteSource(
            name="lab",
            host="example.com",
            user="robot",
            port=2222,
            remote_log_root="/logs/rsl_rl",
            project="project",
            method="scp",
        )
        plan = build_sync_plan(source, cache_root=Path("/tmp/cache"), dry_run=False)

        class FakeCompletedProcess:
            returncode = 23
            stdout = ""
            stderr = "connection failed"

        result = execute_sync_plan(plan, runner=lambda _command, **_kwargs: FakeCompletedProcess())

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.return_code, 23)
        self.assertEqual(result.stderr, "connection failed")


if __name__ == "__main__":
    unittest.main()
