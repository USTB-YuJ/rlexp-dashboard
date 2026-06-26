import unittest
from pathlib import Path


class StaticDashboardTests(unittest.TestCase):
    def test_index_html_contains_run_table_detail_panel_and_api_calls(self):
        html = Path("src/rl_exp_dashboard/web_static/index.html").read_text(encoding="utf-8")

        self.assertIn("id=\"run-table\"", html)
        self.assertIn("id=\"run-detail\"", html)
        self.assertIn("fetch('/api/runs')", html)
        self.assertIn("fetch(`/api/run-detail?run_id=${encodeURIComponent(runId)}`)", html)
        self.assertIn("Metric Summaries", html)
        self.assertIn("Metric Trend", html)
        self.assertIn("id=\"metric-chart\"", html)
        self.assertIn("/api/metric-series", html)
        self.assertIn("renderMetricChart", html)
        self.assertIn("Checkpoints", html)
        self.assertIn("Manual Observation", html)
        self.assertIn("Checkpoint Reviews", html)
        self.assertIn("/api/run-observation", html)
        self.assertIn("/api/checkpoint-review", html)


if __name__ == "__main__":
    unittest.main()
