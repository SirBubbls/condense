"""Neural network model utility functions."""
import numpy as np
from condense.utils.layer_utils import calc_layer_sparsity


def calc_model_sparsity(model):
    """Calculates the average sparsity of a neural network model.

    Args:
      model: keras model
    Returns:
      model sparsity
    """
    return np.average([calc_layer_sparsity(layer) for layer in model.get_weights()],
                      weights=[np.prod(layer.shape) for layer in model.get_weights()])
