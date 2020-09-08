"""Unit prune a neural network layer."""

import numpy as np


def u_prune_layer(ndarray, p_sparsity=0.1, selection=None):
    """Prunes a ndarray to a desired target sparsity.

    Args:
      ndarray: target numpy array
      p_sparsity (float, optional): desired layer sparsity
      selection: neuron selection function
    Returns:
      ndarray: pruned ndarray
    """
    shape = ndarray.shape
    # deep copy layer
    layer = np.array(ndarray)

    neuron_weights = np.abs(layer).sum(0)

    # actual pruning operation
    layer[:, np.argsort(neuron_weights) < int(len(neuron_weights) * p_sparsity)] = 0

    return layer.reshape(shape)
