"""Weight prunes a neural network layer."""
import numpy as np
from condense.optimizer.masking_functions import mask_min_value as default_masking_function


def w_prune_layer(ndarray, p_sparsity=0.1, masking_function=default_masking_function):
    """Weight prunes a ndarray to a desired target sparsity.

    Args:
      ndarray: target numpy array
      p_sparsity (float, optional): desired layer sparsity
      masking_function: neuron masking function
    Returns:
      ndarray: pruned ndarray
    """
    layer = np.array(ndarray)
    layer[masking_function(layer, p_sparsity)] = 0
    return layer
