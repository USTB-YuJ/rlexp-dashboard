import unittest

from rl_exp_dashboard.metrics import sample_scalar_series, summarize_scalar_series


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

    def test_sample_scalar_series_keeps_ordered_evenly_spaced_points(self):
        points = [
            (50, 5.0),
            (0, 0.0),
            (10, 1.0),
            (20, 2.0),
            (30, 3.0),
            (40, 4.0),
        ]

        series = sample_scalar_series("Train/mean_reward", points, max_points=3)

        self.assertEqual(series.tag, "Train/mean_reward")
        self.assertEqual(series.original_count, 6)
        self.assertEqual(series.points, [{"step": 0, "value": 0.0}, {"step": 20, "value": 2.0}, {"step": 50, "value": 5.0}])

    def test_sample_scalar_series_returns_none_for_empty_points(self):
        self.assertIsNone(sample_scalar_series("Train/mean_reward", []))


if __name__ == "__main__":
    unittest.main()
