"""Unit prune a neural network layer."""

import numpy as np
from condense.optimizer.masking_functions import mask_min_value as default_masking_function


def u_prune_layer(ndarray, p_sparsity=0.1, masking_function=default_masking_function):
    """Prunes a ndarray to a desired target sparsity.

    Args:
      ndarray: target numpy array
      p_sparsity (float, optional): desired layer sparsity
      masking_function: neuron masking function
    Returns:
      ndarray: pruned ndarray
    """
    shape = ndarray.shape

    # deep copy layer
    layer = np.array(ndarray)

    neuron_weights = np.abs(layer).sum(0)

    # actual pruning operation
    layer[:, masking_function(neuron_weights, p_sparsity)] = 0

    return layer.reshape(shape)
