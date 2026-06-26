import unittest

from rl_exp_dashboard.metrics import summarize_scalar_series


class MetricSummaryTests(unittest.TestCase):
    def test_summarize_scalar_series_computes_windows_and_slope(self):
        points = [
            (0, 1.0),
            (100, 2.0),
            (200, 3.0),
            (600, 5.0),
            (1100, 8.0),
        ]

        summary = summarize_scalar_series("Train/mean_reward", points, windows=(500, 1000))

        self.assertEqual(summary.tag, "Train/mean_reward")
        self.assertEqual(summary.first_step, 0)
        self.assertEqual(summary.last_step, 1100)
        self.assertEqual(summary.first_value, 1.0)
        self.assertEqual(summary.last_value, 8.0)
        self.assertEqual(summary.min_value, 1.0)
        self.assertEqual(summary.max_value, 8.0)
        self.assertEqual(summary.count, 5)
        self.assertAlmostEqual(summary.window_means["mean500"], 6.5)
        self.assertAlmostEqual(summary.window_means["mean1000"], 4.5)
        self.assertGreater(summary.slope_last_points, 0.0)

    def test_empty_scalar_series_returns_none(self):
        self.assertIsNone(summarize_scalar_series("Train/mean_reward", []))


if __name__ == "__main__":
    unittest.main()
