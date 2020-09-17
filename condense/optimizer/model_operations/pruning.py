"""This module provides a high level interface for tensorflow/keras model pruning."""

import keras
from condense.optimizer.layer_operations.unit_prune import u_prune_layer
from condense.optimizer.layer_operations.weight_prune import w_prune_layer


def prune_model(model, p_sparsity, operations=[w_prune_layer, u_prune_layer], in_place=True):
    """This function prunes a model with a list of given operations.

    Args:
      model: keras model to optimize
      p_sparsity (float): desired model sparsity
      operations (function): layer operations to be performed on each layer
      in_place (boolean): set to true if the original model should be overwritten \
                    if the parameter is set to false a deep copy of the model will be returned.

    Returns:
      model: pruned keras model
    """
    new_weights = []

    for i, layer in enumerate(model.get_weights()):
        # bias values
        if i % 2:
            new_weights.append(layer)
            continue

        # weights
        new_weights.append(w_prune_layer(layer, p_sparsity))

    if in_place:
        model.set_weights(new_weights)
        return model

    c_model = keras.models.clone_model(model)
    c_model.set_weights(new_weights)
    return c_model
