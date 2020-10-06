import pytest
import condense
import numpy as np
import keras


@pytest.fixture
def demo_model():
    model = keras.Sequential(layers=[
        keras.layers.Conv2D(16, (4, 4), input_shape=(20, 20, 3)),
        keras.layers.Conv2D(3, (2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(2)
    ])
    model.compile('adam', 'mse')
    return model


def test_simple_pruning(demo_model):
    def generator():
        while True:
            yield np.random.rand(1, 20, 20, 3), np.random.rand(1, 2)

    gen = generator()
    demo_model.fit(gen, epochs=2, steps_per_epoch=1)
    old_weight = demo_model.layers[1].kernel.numpy()

    pruned = condense.keras.wrap_model(demo_model, condense.optimizer.sparsity_functions.Constant(0.3))
    pruned.compile('adam', 'mse')
    assert (old_weight == pruned.layers[1].layer.kernel.numpy()).all(), 'pruning shouldn\'t change kernel'
