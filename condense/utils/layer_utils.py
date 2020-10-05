"""Neural network layer utility functions."""
import numpy as np


def calc_layer_sparsity(ndarray):
    """This function calculates the sparsity percentage of a tensor.

    Args:
      ndarray: numpy ndarray
    Returns:
      float: sparsity percentage (0.0 - 1.0)
    """
    if not isinstance(ndarray, np.ndarray):
        raise ValueError(f'ndarray required {type(ndarray)} passed: {ndarray}')
    return float(1 - (np.count_nonzero(ndarray) / np.prod(ndarray.shape)))
