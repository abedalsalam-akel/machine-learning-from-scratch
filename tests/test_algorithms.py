"""Deterministic numerical tests covering actual algorithm behavior."""
import numpy as np
import pytest
from ml_from_scratch import hw1, hw2, hw3, hw4, hw5, hw6


def test_regression_gradient_descent_improves_loss():
    x = hw1.add_bias_feature(np.arange(8, dtype=float).reshape(-1, 1))
    y = 2 + 3 * x[:, 1]
    final, history = hw1.gradient_descent(x, y, np.zeros(x.shape[1]), 0.01, 2000)
    assert len(history) == 2000
    assert history[-1] < history[0]
    np.testing.assert_allclose(final, [2, 3], atol=0.03)


def test_pseudoinverse_full_rank_and_rank_deficient():
    x = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([2., 5., 8.])
    np.testing.assert_allclose(hw1.compute_pinv(x, y), [2., 3.], atol=1e-8)
    dependent = np.column_stack([x, x[:, 1]])
    np.testing.assert_allclose(dependent @ hw1.compute_pinv(dependent, y), y, atol=1e-8)


def test_knn_rejects_invalid_k_and_averages_neighbors():
    x = np.array([[0.], [2.], [10.]])
    y = np.array([0., 4., 20.])
    assert np.isclose(hw1.knn_predict(x, y, np.array([[1.]]), 2)[0], 2.0)
    for k in (0, -1, 4):
        with pytest.raises(ValueError):
            hw1.knn_predict(x, y, np.array([[1.]]), k)


def test_decision_tree_perfect_split():
    data = np.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]])
    tree = hw2.DecisionTree(data, hw2.calc_gini, max_depth=3)
    tree.build_tree()
    assert np.isclose(tree.calc_accuracy(data), 1.0)


def test_logistic_regression_learns_simple_binary_dataset():
    x = hw3.add_bias_term(np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]]))
    y = np.array([0, 0, 0, 1, 1, 1])
    model = hw3.LogisticRegressionGD(learning_rate=0.1, max_iter=1000).fit(x, y)
    np.testing.assert_array_equal(model.predict(x), y)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_svm_sample_shapes():
    x, y = hw4.sample_data(2, [[0., 0.], [3., 3.]], 5, 6, seed=3)
    assert x.shape == (11, 2)
    assert set(y) == {-1, 1}


def test_poisson_and_gmm_density():
    assert np.isclose(hw5.poisson_log_pmf(0, 2.), -2.)
    x = np.array([0., 1., -1.])
    normal = hw6.norm_pdf(x, 0., 1.)
    mixed = hw6.gmm_pdf(x, [1.], [0.], [1.])
    assert normal.shape == (3, 1)
    np.testing.assert_allclose(normal, mixed)


def test_kmeans_cluster_centers_and_wcss():
    x = np.array([[0., 0., 0.], [0.1, 0., 0.], [10., 10., 10.], [10.1, 10., 10.]])
    centers, labels, wcss = hw6.kmeans(x, 2, p=2, max_iter=50, random_state=1)
    assert centers.shape == (2, 3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert wcss < 0.02
