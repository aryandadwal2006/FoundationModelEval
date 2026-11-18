import unittest

import numpy as np

from fmeval.forecast import forecast_metrics, make_windows
from fmeval.retrieval import ranking_metrics


class ForecastTests(unittest.TestCase):
    def test_window_construction(self):
        values = np.arange(12, dtype=np.float32)
        x, y = make_windows(values, context=4, horizon=2, start=4, stop=12, stride=2)
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(y.shape, (3, 2))
        np.testing.assert_array_equal(x[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(y[0], [4, 5])

    def test_forecast_metrics(self):
        target = np.array([[1.0, 2.0], [3.0, 4.0]])
        prediction = np.array([[1.0, 1.0], [2.0, 4.0]])
        metrics = forecast_metrics(target, prediction)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.5))


class RetrievalTests(unittest.TestCase):
    def test_ranking_metrics(self):
        metrics = ranking_metrics(["a", "b", "c"], ["b"])
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.5)

    def test_missing_relevant_item(self):
        metrics = ranking_metrics(["a", "c"], ["b"])
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_5"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
