"""This module implements masking functions."""
import numpy as np


def mask_min_value(ndarray, target_sparsity):
    """Default selection algorithm.

    Args:
      ndarray: target numpy array
      target_sparsity: masking intensity
    Returns:
      ndarray: sparsity mask
    """
    shape = ndarray.shape
    ndarray = np.abs(ndarray.flatten())
    thres = np.sort(ndarray)[int(len(ndarray) * target_sparsity)]
    return (ndarray < thres).reshape(shape)
