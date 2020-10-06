"""This module provides a high level interface for tensorflow/keras model pruning."""

import keras
from numpy import linspace
from condense.optimizer.layer_operations.unit_prune import u_prune_layer
from condense.optimizer.layer_operations.weight_prune import w_prune_layer
from condense.utils.layer_utils import calc_layer_sparsity


def prune_model(model, p_sparsity, operations=[w_prune_layer, u_prune_layer], in_place=True):
    """This function prunes a model with a list of given operations.

    Args:
      model: keras model to optimize
      p_sparsity (float or [float]): desired model sparsity as a float or a list of floats. \
                    You can specify the desired sparsity of each layer as a list of floats. \
                    Keep in mind, that you need to specify sparsity for weights and biases seperatly. \
                    Example of a 2 layer neural network: \
                    [l1_bias_sparsity, l1_weight_sparsity, l2_bias_spasity, l2_weight_sparsity] \
                    If you only pass a single float as an argument this target will be used for all layers. \
      operations ([function]): layer operations to be performed on each layer
      in_place (boolean): set to true if the original model should be overwritten \
                    if the parameter is set to false a deep copy of the model will be returned.

    Returns:
      model: pruned keras model
    """
    new_weights = []
    skip = len(model.get_weights()) - 2
    # sparsity_delta = p_sparsity - base_sparsity

    if isinstance(p_sparsity, float):
        p_sparsity = [p_sparsity] * len(model.get_weights())

    if not isinstance(p_sparsity, list):
        raise Exception('p_sparsity is not a float or an list of floats')

    # if sparsity_delta < 0:
        # raise Exception(f'model already satisfies or exceeds target sparsity by {abs(sparsity_delta)}')

    for i, (layer, sparsity) in enumerate(zip(model.get_weights(), p_sparsity)):
        layer_base_sparsity = calc_layer_sparsity(layer)

        if sparsity > 1 or sparsity < 0:
            raise Exception(f'sparsity values have to be between 0.0 and 1.0 (you specified {sparsity})')

        if i > skip:
            new_weights.append(layer)
            continue

        # bias values
        if len(layer.shape) == 1:
            new_weights.append(layer)
            continue

        # splitting up target sparsity in a set of chunks for each operation to satisfy
        pruning_steps = list(linspace(layer_base_sparsity, sparsity, 3))[1:]

        # weight pruning
        weights = layer
        weights = w_prune_layer(weights, pruning_steps.pop(0))
        weights = u_prune_layer(layer, pruning_steps.pop(0))
        new_weights.append(weights)

    if in_place:
        model.set_weights(new_weights)
        return model

    c_model = keras.models.clone_model(model)
    c_model.set_weights(new_weights)
    return c_model
