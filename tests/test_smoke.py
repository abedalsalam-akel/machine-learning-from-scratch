import numpy as np
from ml_from_scratch import hw1, hw2, hw4, hw5, hw6


def test_bias_and_mse():
    x = np.array([[2.], [4.]])
    xb = hw1.add_bias_feature(x)
    np.testing.assert_array_equal(xb, [[1., 2.], [1., 4.]])
    assert np.isclose(hw1.compute_mean_squared_error(xb, np.array([3., 5.]), np.array([1., 1.])), 0)


def test_knn():
    x = np.array([[0.], [1.], [10.]])
    y = np.array([0, 0, 1])
    result = hw1.knn_predict(x, y, np.array([[0.1], [9.9]]), 1)
    np.testing.assert_array_equal(np.asarray(result).reshape(-1), [0, 1])


def test_impurity():
    pure = np.array([[1, 'a'], [2, 'a']], dtype=object)
    mixed = np.array([[1, 'a'], [2, 'b']], dtype=object)
    assert np.isclose(hw2.calc_entropy(pure), 0.)
    assert np.isclose(hw2.calc_gini(mixed), 0.5)


def test_svm_sample_data():
    x, y = hw4.sample_data(2, [[0, 0], [2, 2]], 4, 5, seed=3)
    assert x.shape == (9, 2)
    np.testing.assert_array_equal(y, [-1] * 4 + [1] * 5)


def test_poisson_logpmf():
    assert np.isclose(hw5.poisson_log_pmf(0, 2.), -2.)
    assert np.isclose(hw5.poission_analytic_mle(np.array([1., 2., 3.])), 2.)


def test_initial_centroids():
    x = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    c = hw6.get_random_centroids(x, 2, random_state=7)
    assert c.shape == (2, 3)
    assert all(any(np.array_equal(v, row) for row in x) for v in c)
