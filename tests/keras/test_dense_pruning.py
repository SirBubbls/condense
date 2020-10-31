import pytest
import condense
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from logging import info
from condense.utils.model_utils import calc_model_sparsity
from condense.utils.layer_utils import calc_layer_sparsity
import tensorflow_datasets as tfds


@pytest.fixture
def model():
    """Prepare test keras model."""
    model = Sequential(layers=[
        Dense(256, input_shape=(None, 10)),
        Dense(128),
        Dense(16)
    ])
    model.compile(keras.optimizers.Adam(learning_rate=0.0), 'mse')
    return model


def test_simple_model(model):
    def generator():
        while True:
            yield np.random.rand(10).reshape(1, -1), np.random.rand(16)

    pruned = condense.keras.wrap_model(model, condense.optimizer.sparsity_functions.Linear(0.8))

    assert pruned
    assert len(pruned.layers) == len(model.layers), 'layer count should not change due to wrapping'
    assert model.layers[1].name in pruned.layers[1].name, 'original layer name should be in the wrapper name'
    assert model.layers[1].name != pruned.layers[1].name, 'wrapped layers should be displayed as wrapped'
    for layer1, layer2 in zip(model.layers, pruned.layers):
        assert layer1 is not layer2, 'wrapped layers should be displayed as wrapped'
    gen = generator()
    model.compile('adam', 'mse')
    assert model.fit(gen, epochs=1, steps_per_epoch=10), 'fit generator on original model'
    info('Original Dense Model Sparsity: %f', calc_model_sparsity(model))
    pruned.compile('adam', 'mse')
    assert pruned.fit(gen,
                      epochs=1,
                      steps_per_epoch=10,
                      callbacks=[condense.keras.callbacks.PruningCallback()]), 'fit generator on pruned model'


def test_pruning_accuracy(model):
    def _test_layer_pruning(target):
        info(f'## Testing Target {target} ##')

        weights = model
        info("Base Model Sparsity %f", calc_model_sparsity(weights))
        pruned = condense.keras.wrap_model(model, condense.optimizer.sparsity_functions.Constant(target))
        base_layer_sparsity = [calc_layer_sparsity(layer.kernel.numpy()) for layer in model.layers]
        info(f'Layer Base Sparsity Values {base_layer_sparsity}')

        pruned.compile(keras.optimizers.Adam(learning_rate=0.0), 'mse')
        pruned.fit(x=np.random.randn(1, 10),
                   y=np.random.randn(1, 16),
                   callbacks=[condense.keras.callbacks.PruningCallback()])

        for layer, old_layer_sparsity in zip(pruned.layers, base_layer_sparsity):
            if isinstance(layer, condense.keras.PruningWrapper):
                info(calc_layer_sparsity(layer.layer.kernel.numpy()))
                new_layer_sparsity = calc_layer_sparsity(layer.layer.kernel.numpy())
                assert abs(target - new_layer_sparsity) < 0.1, 'Layer sparsity target delta to high'
            else:
                info('skipped')

    for target in np.linspace(0.1, 0.95, 4):
        _test_layer_pruning(target)


def test_model_construction():
    model = Sequential(layers=[
        Dense(10, input_shape=(None, 10)),
        Dense(10),
        Dense(16)
    ])
    model.compile(keras.optimizers.SGD(learning_rate=0), 'mse')
    model.build()

    old_layer_weights = [layer.kernel.numpy() for layer in model.layers]

    pruned = condense.keras.wrap_model(model, condense.optimizer.sparsity_functions.Linear(0.2))

    for old_weights, pruned_layer in zip(old_layer_weights, pruned.layers):
        if not isinstance(pruned_layer, condense.keras.PruningWrapper):
            continue
        assert (old_weights == pruned_layer.layer.kernel.numpy()).all(), 'weights should be unchanged after wrapping'


def test_kernel_manipulation(model):
    layer_weight = np.array(model.layers[2].kernel.numpy())

    pruned = condense.keras.wrap_model(model, condense.optimizer.sparsity_functions.Constant(0.00))
    pruned.build()
    pruned.compile(keras.optimizers.SGD(learning_rate=0), 'mse')

    pruned.fit(x=np.random.randn(1, 10),
               y=np.random.randn(1, 16),
               callbacks=[condense.keras.callbacks.PruningCallback()])

    diff = layer_weight - pruned.layers[2].layer.kernel.numpy()
    # TODO should actually be == 0
    assert np.count_nonzero(diff) <= 1, 'kernel shouldn\'t change with 0% pruning'


def test_iris_pruning():
    # Loading Dataset
    ds = tfds.load('iris', split='train', shuffle_files=True, as_supervised=True)
    assert ds, 'datasets could not be loaded'

    # Loading Iris Model
    model = keras.models.load_model('tests/iris.h5', compile=False)
    model.compile('adam', 'categorical_crossentropy')
    assert model, 'iris model could not be loaded'

    # Evaluate old model
    acc_old = model.evaluate(ds.batch(20), steps=2)
    info(f'Iris Base Accuracy: {acc_old}')

    pruned = condense.keras.wrap_model(model, condense.optimizer.sparsity_functions.Constant(0.3))
    pruned.compile('adam', 'mse')
    pruned.fit(ds.batch(20), epochs=1, callbacks=[condense.keras.PruningCallback()])

    # Evaluate new model
    acc_new = pruned.evaluate(ds.batch(20), steps=2)
    info(f'Pruned Iris Accuracy: {acc_new}')

    assert abs(acc_new - acc_old) < 0.15, 'more than 15% acc loss by pruning 30%'
