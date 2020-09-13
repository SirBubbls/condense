"""Tests for layer utility functions."""
import numpy as np
import condense.utils.layer_utils as lu


def test_calc_layer_sparsity():
    """Testing layer sparsity calculation."""
    test_ndarray = np.array([[0, 2, 0], [1, 0, 1]])
    assert lu.calc_layer_sparsity(test_ndarray) == 3 / 6, 'correct sparsity value'

    test_ndarray = np.array([[0, 0, 0], [1, 0, 1]])
    assert abs(lu.calc_layer_sparsity(test_ndarray) - 4 / 6) < 10**-8, 'correct sparsity value'
    assert lu.calc_layer_sparsity(np.zeros((20, 20))) == 1.0, 'zero array should have 1.0 sparsity'
    assert lu.calc_layer_sparsity(
        np.random.rand(20, 20)) == 0.0, 'random array should have 0.0 sparsity'
