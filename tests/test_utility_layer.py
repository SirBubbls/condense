"""Tests for layer utility functions."""
import numpy as np
import condense.utils.layer_utils as lu


def test_calc_layer_sparsity():
    """Testing layer sparsity calculation"""
    test_ndarray = np.array([[0, 2, 0], [1, 0, 1]])
    assert lu.calc_layer_sparsity(test_ndarray) == 3 / 6, 'correct sparsity value'
    test_ndarray = np.array([[0, 0, 0], [1, 0, 1]])
    assert lu.calc_layer_sparsity(test_ndarray) == 2 / 6, 'correct sparsity value'
