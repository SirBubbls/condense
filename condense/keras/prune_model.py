"""This module provides a high level interface for layer augmentation."""
import logging
from keras.models import clone_model
from condense.keras import wrappers
from condense.keras import support
from condense.optimizer import sparsity_functions
from copy import deepcopy


def wrap_model(model, sparsity_fn):
    """This function turns a model into a prunable copy of itself.

    Args:
      model: Target model
      sparsity_fn: desired sparsity function for this model
    Todos:
       * layers are not deep copied
       * support for custom weights in PruningWrapper
    Returns:
      Augemented model (not a deepcopy)
    """
    if not issubclass(type(sparsity_fn), sparsity_functions.SparsityFunction):
        raise ValueError("""argument sprasity_fn should be a subclass of SparsityFunction.""")

    class __WrappingFunction:
        def __init__(self, sparsity_fn):
            self.funciton = sparsity_fn

        def wrap(self, layer):
            if not support.is_supported_layer(layer) or 'output' in layer.name:
                logging.warning('Layer %s is not supported.', layer.get_config()["name"])
                return layer
            wrapper = wrappers.PruningWrapper(layer, deepcopy(sparsity_fn))
            return wrapper

    # It is important to get the weights of each layer individually,
    # because the wrapper will add additional variables to the model.
    weights = [layer.get_weights() for layer in model.layers]

    temp_wrapper = __WrappingFunction(sparsity_fn)
    new_model = clone_model(model=model,
                            clone_function=temp_wrapper.wrap)

    # Apply saved weights to each layer of the wrapped model individually.
    for weight, layer in zip(weights, new_model.layers):
        if isinstance(layer, wrappers.PruningWrapper):
            layer.layer.set_weights(weight)

    if model.optimizer and model.loss:
        new_model.compile(model.optimizer, model.loss)

    return new_model


def wrap_layer(layer, sparsity_fn):
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
    return wrappers.PruningWrapper(layer, sparsity_fn)
