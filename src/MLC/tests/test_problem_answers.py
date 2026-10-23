"""Regression tests for the one-file-per-problem reference answers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

MLC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MLC_ROOT))

from problems.classic_ml.binary_metrics_roc_auc import (  # noqa: E402
    binary_classification_metrics,
    roc_auc,
)
from problems.classic_ml.conv2d import conv2d_valid  # noqa: E402
from problems.classic_ml.decision_tree_split import best_gini_split  # noqa: E402
from problems.classic_ml.kmeans import kmeans  # noqa: E402
from problems.classic_ml.knn import knn_predict  # noqa: E402
from problems.classic_ml.linear_regression import (  # noqa: E402
    linear_regression_gradient_descent,
)
from problems.classic_ml.logistic_regression import (  # noqa: E402
    logistic_regression_gradient_descent,
)
from problems.classic_ml.pca import principal_component_analysis  # noqa: E402
from problems.classic_ml.reservoir_sampling import reservoir_sample  # noqa: E402
from problems.classic_ml.softmax_cross_entropy import (  # noqa: E402
    cross_entropy_from_logits,
    softmax,
)
from problems.language_models.scaled_dot_product_attention import (  # noqa: E402
    scaled_dot_product_attention,
)
from problems.language_models.tfidf import tfidf  # noqa: E402


class ProblemAnswersTest(unittest.TestCase):
    def test_softmax_and_cross_entropy_are_stable(self):
        logits = np.array([[1_000.0, 1_001.0], [-1_000.0, -999.0]])
        probabilities = softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(cross_entropy_from_logits(logits, np.array([1, 1]))))

    def test_linear_regression_recovers_line(self):
        x = np.linspace(-1.0, 1.0, 40)[:, None]
        y = 3.0 * x[:, 0] - 2.0
        weights, bias = linear_regression_gradient_descent(x, y, steps=2_000)
        np.testing.assert_allclose(weights, [3.0], atol=1e-3)
        self.assertAlmostEqual(bias, -2.0, places=3)

    def test_logistic_regression_separates_simple_data(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        weights, bias = logistic_regression_gradient_descent(x, y, steps=1_000)
        predictions = (x @ weights + bias >= 0).astype(int)
        np.testing.assert_array_equal(predictions, y)

    def test_knn(self):
        train_x = np.array([[0.0], [1.0], [9.0], [10.0]])
        train_y = np.array([0, 0, 1, 1])
        predictions = knn_predict(train_x, train_y, np.array([[0.2], [9.8]]), k=1)
        np.testing.assert_array_equal(predictions, [0, 1])

    def test_kmeans(self):
        x = np.array([[0.0], [0.2], [10.0], [10.2]])
        centers, labels = kmeans(x, 2, seed=1)
        np.testing.assert_allclose(np.sort(centers[:, 0]), [0.1, 10.1], atol=1e-6)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_best_gini_split(self):
        split = best_gini_split(
            np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1])
        )
        self.assertIsNotNone(split)
        assert split is not None
        self.assertEqual(split[0], 0)
        self.assertAlmostEqual(split[1], 1.5)
        self.assertAlmostEqual(split[2], 0.0)

    def test_best_gini_split_with_large_finite_features(self):
        features = np.array([[1.0e308], [1.2e308]])
        split = best_gini_split(features, np.array([0, 1]))
        self.assertIsNotNone(split)
        assert split is not None
        self.assertTrue(np.isfinite(split[1]))
        self.assertGreater(split[1], features[0, 0])
        self.assertLess(split[1], features[1, 0])
        self.assertAlmostEqual(split[2], 0.0)

        crossing_zero = np.array([[-1.7e308], [1.7e308]])
        crossing_split = best_gini_split(crossing_zero, np.array([0, 1]))
        self.assertIsNotNone(crossing_split)
        assert crossing_split is not None
        self.assertTrue(np.isfinite(crossing_split[1]))
        self.assertAlmostEqual(crossing_split[2], 0.0)

    def test_pca(self):
        x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        transformed, components, explained_variance = principal_component_analysis(x, 1)
        self.assertEqual(transformed.shape, (3, 1))
        self.assertEqual(components.shape, (1, 2))
        self.assertGreater(explained_variance[0], 0.0)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)

    def test_valid_convolution(self):
        image = np.arange(1, 10).reshape(3, 3)
        kernel = np.array([[1, 0], [0, -1]])
        np.testing.assert_allclose(conv2d_valid(image, kernel), -4.0)

    def test_masked_attention(self):
        q = np.array([[1.0, 0.0]])
        k = np.array([[1.0, 0.0], [0.0, 1.0]])
        v = np.array([[2.0], [8.0]])
        output, weights = scaled_dot_product_attention(q, k, v, [[True, False]])
        np.testing.assert_allclose(output, [[2.0]])
        np.testing.assert_allclose(weights, [[1.0, 0.0]])

    def test_metrics_and_auc(self):
        metrics = binary_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)

    def test_reservoir_sample(self):
        sample = reservoir_sample(range(100), 10, seed=7)
        self.assertEqual(len(sample), 10)
        self.assertEqual(len(set(sample)), 10)
        self.assertTrue(set(sample).issubset(set(range(100))))

    def test_tfidf(self):
        matrix, vocabulary = tfidf(["red blue red", "blue green"])
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(vocabulary, ["blue", "green", "red"])
        self.assertGreater(matrix[0, vocabulary.index("red")], 0.0)


if __name__ == "__main__":
    unittest.main()

