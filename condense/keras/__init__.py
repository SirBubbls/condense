"""This module implements pruning functionality in keras models."""

import condense.keras.wrappers as wrappers
from condense.keras.callbacks import PruningCallback
from keras.models import Sequential


def prune_model(model):
    """This function makes a model prunable.

    Args:
      model: Target model
    Returns:
      augemented model
    """
    return Sequential(layers=[wrappers.PruningWrapper(layer) for layer in model.layers])
