import unittest
from pathlib import Path


class StaticDashboardTests(unittest.TestCase):
    def test_index_html_contains_run_table_detail_panel_and_api_calls(self):
        html = Path("src/rl_exp_dashboard/web_static/index.html").read_text(encoding="utf-8")

        self.assertIn("id=\"run-table\"", html)
        self.assertIn("id=\"run-search\"", html)
        self.assertIn("id=\"project-filter\"", html)
        self.assertIn("id=\"group-filter\"", html)
        self.assertIn("id=\"run-sort\"", html)
        self.assertIn("renderFilteredRuns", html)
        self.assertIn("id=\"run-detail\"", html)
        self.assertIn("fetch('/api/runs')", html)
        self.assertIn("Project Configs", html)
        self.assertIn("id=\"project-configs\"", html)
        self.assertIn("fetch('/api/projects')", html)
        self.assertIn("renderProjectConfigs", html)
        self.assertIn("Remote Sources", html)
        self.assertIn("id=\"remote-sources\"", html)
        self.assertIn("fetch('/api/remote-sources')", html)
        self.assertIn("renderRemoteSources", html)
        self.assertIn("Include Patterns", html)
        self.assertIn("Exclude Patterns", html)
        self.assertIn("Lineage Overview", html)
        self.assertIn("id=\"lineage-overview\"", html)
        self.assertIn("fetch('/api/lineage')", html)
        self.assertIn("renderLineageOverview", html)
        self.assertIn("fetch(`/api/run-detail?run_id=${encodeURIComponent(runId)}`)", html)
        self.assertIn("Metric Summaries", html)
        self.assertIn("Metric Trend", html)
        self.assertIn("id=\"metric-chart\"", html)
        self.assertIn("/api/metric-series", html)
        self.assertIn("renderMetricChart", html)
        self.assertIn("Git Metadata", html)
        self.assertIn("gitPanel", html)
        self.assertIn("Checkpoints", html)
        self.assertIn("Videos", html)
        self.assertIn("Artifacts", html)
        self.assertIn("artifactTable", html)
        self.assertIn("videoTable", html)
        self.assertIn("/api/artifact-file", html)
        self.assertIn("artifactFileUrl", html)
        self.assertIn("<video", html)
        self.assertIn("Open", html)
        self.assertIn("Manual Observation", html)
        self.assertIn("Checkpoint Reviews", html)
        self.assertIn("/api/run-observation", html)
        self.assertIn("/api/checkpoint-review", html)

    def test_index_html_contains_compare_runs_panel_and_api_call(self):
        html = Path("src/rl_exp_dashboard/web_static/index.html").read_text(encoding="utf-8")

        self.assertIn("Compare Runs", html)
        self.assertIn("id=\"compare-before-run\"", html)
        self.assertIn("id=\"compare-after-run\"", html)
        self.assertIn("id=\"compare-result\"", html)
        self.assertIn("/api/compare", html)
        self.assertIn("before_run_id", html)
        self.assertIn("after_run_id", html)
        self.assertIn("renderCompareResult", html)
        self.assertIn("Config Diffs", html)
        self.assertIn("Metric Deltas", html)


if __name__ == "__main__":
    unittest.main()
