"""This module implements a use case for the lottery ticket hypothesis.

For a detailed explanation of this hypothesis check out https://arxiv.org/abs/1803.03635.
There is also an implementation available at https://github.com/google-research/lottery-ticket-hypothesis.
"""
import condense
import numpy as np
import tensorflow as tf
import keras
from copy import deepcopy
from condense.optimizer.sparsity_functions import Constant
from . import PruningCallback, PruningWrapper
from .prune_model import wrap_model


def find_winning_ticket(model, dataset, sparsity):
    """This function tries to find a winning ticket of a model.

    Args:
      model: keras model
      dataset: dataset generator
      sparsity: desired target sparsity

    Returns:
      weights for initialization
      sparsity masks for each layer
      wrapped model
    """

    # we need to save the original weights
    wrapped = wrap_model(model, Constant(sparsity))
    wrapped.compile('adam', 'mse')

    condense.logger.info('### Searching for winning ticket  ###')
    hist = wrapped.fit(dataset, epochs=20, steps_per_epoch=1, callbacks=[PruningCallback()], verbose=0)
    condense.logger.info('### Finished searching  ###')

    # Exctract sparsity mask of each layer
    masks = []
    for layer in wrapped.layers:
        if isinstance(layer, PruningWrapper):
            masks.append(deepcopy(layer.mask.numpy()))
        else:
            masks.append(None)

    return hist, masks, wrapped


class Trainer():
    """Helper class for training operations."""
    def __init__(self, model, t_sparsity):
        """
        """
        self.training_parameters = {}

        self.base_model = model
        self.base_model.build()

        self.init_weights = [deepcopy(layer.get_weights()) for layer in model.layers]

        self.mask = None
        self.training_model = None
        self.t_sparsity = t_sparsity
        # Default Optimizer
        self.optimizer = keras.optimizers.SGD(learning_rate=0.001)
        self.loss = 'mse'
        self.history = {}

    def train(self, dataset, epochs, steps_per_epoch=1, eval_data=None):
        """This function runs the complete training process.

        Args:
          dataset: data generator for training
          epochs: training epochs
          steps_per_epoch: steps per epoch
          eval_data: generator for evaluation data
        """

        # Finding winning ticket
        hist, mask, wrapped_model = find_winning_ticket(self.base_model,
                                                     dataset,
                                                     self.t_sparsity)
        self.history['ticket_search'] = hist

        # Apply mask
        self.mask = mask
        condense.logger.info('Winning ticket found')

        condense.logger.info('Resetting model to initial parameters')
        self.__reset(wrapped_model)
        condense.logger.info('Remove remaining weights')
        self.__remove_remaining(wrapped_model)

        condense.logger.info('Recompile Model')
        wrapped_model.compile(self.optimizer, self.loss)

        condense.logger.info('Training on pruned model')
        self.training_model = wrapped_model
        self.history['training'] = wrapped_model.fit(dataset,
                                                     epochs=epochs,
                                                     steps_per_epoch=steps_per_epoch,
                                                     validation_data=eval_data,
                                                     validation_steps=2,
                                                     workers=1)
        return self.history['training']

    def __reset(self, model):
        """ """
        for layer, weights in zip(model.layers, self.init_weights):
            if not isinstance(layer, PruningWrapper):
                continue
            condense.logger.info(f'Resetting layer {layer.name}')
            layer.layer.set_weights(weights)

    def __remove_remaining(self, model):
        for layer in model.layers:
            if isinstance(layer, PruningWrapper):
                assert ((layer.kernel.numpy() != 0) == (layer.mask.numpy() == 1)).all(), 'mask check'
