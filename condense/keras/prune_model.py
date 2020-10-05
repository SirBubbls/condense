"""This module provides a high level interface for layer augmentation."""
import logging
from keras.models import clone_model
from condense.keras import wrappers
from condense.keras import support
from condense.optimizer import pruning_strategies


def wrap_model(model):
    """This function turns a model into a prunable copy of itself.

    Args:
      model: Target model
    Todos:
       * layers are not deep copied
    Returns:
      Augemented model (not a deepcopy)
    """
    return clone_model(model=model,
                       clone_function=wrap_layer)


def wrap_layer(layer):
    """This function applies the PruningWrapper class to a the target layer if pruning is supported.

    The main use for this function is to serve as a clone_function for keras.models.clone_model().

    Args:
      layer: Keras Layer to be wrapped.

    Returns:
      Either a wrapped layer if supported or the original layer.
    """
    if not support.is_supported_layer(layer):
        logging.warning('Layer %s is not supported.', layer.get_config()["name"])
        return layer
    return wrappers.PruningWrapper(layer, pruning_strategies.Linear(0.75))
