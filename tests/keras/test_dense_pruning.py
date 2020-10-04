import condense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from logging import info
from condense.utils.model_utils import calc_model_sparsity


def test_simple_model():
    model = Sequential(layers=[
        Dense(20, input_shape=(4, )),
        Dense(30),
        Dense(30),
        Dense(2)
    ])

    def generator():
        while True:
            yield np.random.rand(4).reshape(1, -1), np.random.rand(2)

    pruned = condense.keras.wrap_model(model)

    assert pruned
    assert len(pruned.layers) == len(model.layers), 'layer count should not change due to wrapping'
    assert model.layers[1].name in pruned.layers[1].name, 'original layer name should be in the wrapper name'
    assert model.layers[1].name != pruned.layers[1].name, 'wrapped layers should be displayed as wrapped'
    for layer1, layer2 in zip(model.layers, pruned.layers):
        assert layer1 is not layer2, 'wrapped layers should be displayed as wrapped'
    gen = generator()
    model.compile('adam', 'mse')
    assert model.fit(gen, epochs=1, steps_per_epoch=10), 'fit generator on original model'
    info('Original Dense Model Sparsity: %f', sparse_old := calc_model_sparsity(model))
    pruned.compile('adam', 'mse')
    assert pruned.fit(gen,
                      epochs=1,
                      steps_per_epoch=10,
                      callbacks=[condense.keras.callbacks.PruningCallback()]), 'fit generator on pruned model'
    info('Pruned Dense Model Sparsity: %f', calc_model_sparsity(pruned))
    # assert calc_model_sparsity(model) == sparse_old, 'base model sparsity changed after pruning'
    assert calc_model_sparsity(pruned) > sparse_old + .15, 'sparsity didn\'t increase over 15%'
