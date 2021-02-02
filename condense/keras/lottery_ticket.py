"""This module implements a use case for the lottery ticket hypothesis.

For a detailed explanation of this hypothesis check out https://arxiv.org/abs/1803.03635.
There is also an implementation available at https://github.com/google-research/lottery-ticket-hypothesis.
"""
from copy import deepcopy

import numpy as np

import condense
from condense.optimizer.sparsity_functions import Constant

from . import PruningCallback, PruningWrapper
from .prune_model import wrap_model


def find_winning_ticket(model, dataset, validation_data=None):
    """This function tries to find a winning ticket of a model.

    Args:
      model: keras model
      dataset: dataset generator
      sparsity: desired target sparsity

    Returns:
      history of .fit operation
      sparsity masks for each layer
    """
    hist = model.fit(dataset,
                     epochs=30,
                     steps_per_epoch=2,
                     callbacks=[PruningCallback()],
                     verbose=0,
                     validation_data=validation_data)

    # Exctract sparsity mask of each layer
    masks = []
    for layer in model.layers:
        if isinstance(layer, PruningWrapper):
            masks.append(deepcopy(layer.mask.numpy()))
        else:
            masks.append(None)

    return hist, masks


class Trainer():
    """Helper class for training operations.

    Attributes:
        mask ([numpy.ndarray]): A list of numpy arrays.
                                Each entry correspondes to the mask of a layer.
        t_sparsity (float): Target sparsity for each layer.
        optimizer (keras.optimizer.Optimizer): Optimizer for training (defaults to SGD).
        loss: Loss function.
        history (dict): A dict with the results of each training/optimization step.
        training_model (keras.models.Model): model for training
    """
    def __init__(self, model, t_sparsity):
        """Constructor.

        Args:
           model (keras.models.Model): An unpruned keras model
           t_sparsity (float): desired sparsity of each layer
        """
        # Default or model optimizer & loss
        if not model.optimizer or not model.loss:
            raise Exception('model is not compiled. please compile your model first.')

        self.model = model

        # Build and save inital model parameters
        self.model.build()
        self.init_weights = condense.keras.wrappers.get_internal_weights(self.model)

        self.mask = []
        self.t_sparsity = t_sparsity
        self.history = {}
        self.training_model = None

    def train(self, dataset, epochs, steps_per_epoch=1, eval_data=None):
        """This function runs the complete training process.

        Args:
          dataset: generator for training batches
          epochs: training epochs
          steps_per_epoch: steps per epoch
          eval_data: generator for evaluation data
        """
        self.training_model = self.model

        # Agument model with PruningWrapper
        if not self.is_model_wrapped(self.training_model):
            condense.logger.warning("""Passed model is not yet wrapped.
            Model will get wrapped automatically. If you want more control over wrapping please call
            condense.keras.wrap_model on your model first and then use this class.""")
            self.training_model = wrap_model(self.training_model,
                                             Constant(self.t_sparsity))

        # Finding winning ticket
        condense.logger.info('Searching for winning ticket')
        hist, mask = find_winning_ticket(self.training_model,
                                         dataset)
        self.history['ticket_search'] = hist

        # Apply mask
        self.mask = mask
        condense.logger.info('Winning ticket found')

        condense.logger.info('Resetting model to initial parameters')
        self._reset(self.training_model)

        condense.logger.info('Training on pruned model')
        self.history['training'] = self.training_model.fit(dataset,
                                                           epochs=epochs,
                                                           steps_per_epoch=steps_per_epoch,
                                                           validation_data=eval_data,
                                                           validation_steps=2,
                                                           verbose=0)
        self.training_model = self.training_model

        return self.history['training']

    @staticmethod
    def is_model_wrapped(model):
        """Returns if model is wrapped or not."""
        for layer in model.layers:
            if isinstance(layer, condense.keras.PruningWrapper):
                return True
        return False

    def _reset(self, model):
        """This function resets the model and its internal layers to its initial weights.

        Keep in mind, that the mask of each PruningWrapper is untouched by this operation.
        """
        for layer, weights in zip(model.layers, self.init_weights):
            if isinstance(layer, PruningWrapper):
                condense.logger.info('Resetting layer %s', layer.name)
                layer.layer.set_weights(weights)
            else:
                layer.set_weights(weights)

    @staticmethod
    def reset_mask(model):
        """Reset all masks of each PruningWrapper in the model.

        Args:
           model (keras.models.Model): target model
        """
        for layer in model.layers:
            if isinstance(layer, condense.keras.PruningWrapper):
                layer.mask.assign(np.ones(layer.mask.shape))
