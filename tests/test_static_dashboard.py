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
        self.assertIn("Checkpoints", html)


if __name__ == "__main__":
    unittest.main()
