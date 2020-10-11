"""Tests lottery ticket."""
import pytest
import condense
import logging
import keras
from keras.layers import Dense
import numpy as np


def _dense_generator():
    while True:
        yield np.random.rand(10).reshape(1, -1), np.random.rand(16)


@pytest.fixture
def dense_model():
    """Prepare test keras model."""
    model = keras.models.Sequential(layers=[
        Dense(256, input_shape=(None, 10)),
        Dense(128),
        Dense(16)
    ])
    model.compile(keras.optimizers.Adam(learning_rate=0.0), 'mse')
    return model


def test_lottery_ticket(dense_model):
    def generator():
        while True:
            yield np.random.rand(10).reshape(1, -1), np.random.rand(16)

    pruned = condense.keras.wrap_model(dense_model, condense.optimizer.sparsity_functions.Linear(0.8))
    pruned.summary()
    assert pruned, 'PruningWrapper wasn\'t applied correctly'
    gen = generator()
    pruned.compile('adam', 'mse')
    assert pruned.fit(gen,
                      epochs=3,
                      steps_per_epoch=10,
                      callbacks=[condense.keras.callbacks.PruningCallback()]), 'fit generator on pruned model'


def test_find_winning_ticket(dense_model):
    generator = _dense_generator()
    dense_model = condense.keras.wrap_model(dense_model,
                                            condense.optimizer.sparsity_functions.Constant(.1))
    hist, masks = condense.keras.find_winning_ticket(dense_model, generator)
    assert hist and masks, 'return values'
    assert len(dense_model.layers) == len(masks)

def test_trainer(dense_model):
    generator = _dense_generator()
    trainer = condense.keras.lottery_ticket.Trainer(dense_model, 0.5)

    hist = trainer.train(generator, 10)
    logging.info(hist.history)

    # Check for masking
    for i, layer in enumerate(trainer.training_model.layers):
        if not isinstance(layer, condense.keras.PruningWrapper):
            continue

        logging.info(f'{layer.name} kernel after training: {layer.layer.kernel.numpy()}')

        assert (layer.mask.numpy() == (layer.kernel.numpy() != 0)).all(), f'sparsity lost {layer.name} ({i})'


def test_trainer_reset(dense_model):
    """This test makes sure, that model parameter resetting works as intended."""
    model = dense_model
    model.compile(keras.optimizers.Adam(learning_rate=1.0), 'mse')
    trainer = condense.keras.Trainer(model, 0.5)

    assert isinstance(trainer.model, keras.Model), 'base model should be present after initialization'
    assert len(trainer.init_weights) == len(trainer.model.layers), 'init weights saved'

    # Fitting should change model weights/parameters
    trainer.model.fit(_dense_generator(), epochs=10, steps_per_epoch=1, verbose=0)

    # Check if weights changed after training
    for i, (layer, init_weights) in enumerate(zip(trainer.model.layers, trainer.init_weights)):
        if isinstance(layer, condense.keras.PruningWrapper):
            layer_weights = layer.layer.get_weights()
        else:
            layer_weights = layer.get_weights()

        for current_w, init_w in zip(layer_weights, init_weights):
            assert (current_w != init_w).any(), f'unchanged weights at layer {i}'

    trainer._reset(trainer.model)

    # Check if weights were reset
    for i, (layer, init_weights) in enumerate(zip(trainer.model.layers, trainer.init_weights)):
        if isinstance(layer, condense.keras.PruningWrapper):
            layer_weights = layer.layer.get_weights()
        else:
            layer_weights = layer.get_weights()

        for current_w, init_w in zip(layer_weights, init_weights):
            assert (current_w == init_w).all(), f'weights at layer {i} not in initial configuration'
