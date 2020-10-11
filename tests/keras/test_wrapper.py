from copy import deepcopy

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

import condense
from condense.keras import PruningWrapper
from condense.optimizer.sparsity_functions import Constant


def test_gradient_generation():
    """This test checks gradient calculation with and without a mask applied to the kernel.

    Steps:
        1. generating one wrapped dense layer (save its initial parameter configuration)
        2. generate input data
        3. forward pass and save gradients from kernel
        4. resetting parameters
        5. prune internal kernel
        6. run forward pass again and calculate gradients from kernel (same input data)
        7. gradients from first and second pass should be equal
    """
    wrapped = PruningWrapper(Dense(5, input_shape=(3,)),
                             condense.optimizer.sparsity_functions.Constant(0.5))
    output = Dense(1)

    # Building both layers
    wrapped.build(input_shape=(3,))
    output.build(input_shape=(5,))

    # Save original weights
    w_wrapped, w_output = deepcopy(wrapped.get_weights()), deepcopy(output.get_weights())

    # Input data generation
    data = np.random.random((1, 3))

    assert (wrapped.layer.kernel.numpy() != 0).all(), 'kernel is masked'

    # Pre Wrapping
    with tf.GradientTape() as tape:
        out = wrapped.call(tf.convert_to_tensor(data))
        final = output.call(out)
        grad_pre = tape.gradient(final, [wrapped.layer.kernel])

    # Resetting Weights
    wrapped.set_weights(w_wrapped)
    output.set_weights(w_output)

    assert (w_wrapped[0] == wrapped.layer.kernel.numpy()).all(), 'weights not resetted'

    # Pruning Operation
    wrapped.prune(0.5)

    # Sanity check kernel
    assert ((wrapped.layer.kernel.numpy() != 0) == wrapped.mask.numpy()).all(), 'mask not applied on kernel'

    # After Wrapping
    with tf.GradientTape() as tape:
        out = wrapped.call(tf.convert_to_tensor(data))
        final = output.call(out)
        grad_after = tape.gradient(final, [wrapped.layer.kernel])

    assert (grad_pre[0].numpy() == grad_after[0].numpy()).all(), 'Gradients shouldn\'t be different'


def test_wrapper_mask_application():
    """Check if mask gets applied correctly to layer."""
    wrapped = PruningWrapper(Dense(5, input_shape=(3,)),
                             condense.optimizer.sparsity_functions.Constant(0.5))
    _ = Dense(1)
    wrapped.build(input_shape=(3,))
    assert not (wrapped.mask.numpy() == 0).any(), 'mask not initalized correctly'
    assert (wrapped.kernel.numpy() == wrapped.layer.kernel.numpy()).all(), """
    PruningWrapper.kernel should return masked kernel"""

    wrapped.prune(0.5)
    assert (wrapped.kernel.numpy() == wrapped.layer.kernel.numpy()).all(), ""


def test_get_internal_weights():
    """Test if get_internal_weights works as intended."""
    model = Sequential(layers=[
        Dense(10, input_shape=(3,)),
        PruningWrapper(Dense(20), Constant(0.5)),
        PruningWrapper(Dense(20), Constant(0.5)),
        Dense(10, input_shape=(3,))
    ])

    model.build()
    internal_weights = condense.keras.wrappers.get_internal_weights(model)

    for layer, weight in zip(model.layers, internal_weights):

        # Check for Bias & Weight
        assert len(weight) == 2, 'layers should have 2 weights'
        assert len(weight[0].shape) == 2, 'weights should have two dimensions'
        assert len(weight[1].shape) == 1, 'bias one dimensional'

        if isinstance(layer, PruningWrapper):
            layer = layer.layer

        assert (layer.kernel.numpy() == weight[0]).all(), 'Kernel doesn\'t match'
        assert (layer.bias.numpy() == weight[1]).all(), 'Bias doesn\'t match'
