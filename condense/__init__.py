"""Condense is a pruning framework for artifical neural networks.

This module provides pruning methods for various artifical neural network architectures.

You can check out the documentation of this module under: https://sirbubbls.github.io/condense
Open source repository: github.com/SirBubbls/condense
"""

import sys
import condense.optimizer
import condense.utils
import condense.keras


if sys.version_info < (3,):
    raise Exception("Python 2 has reached end-of-life and is no longer supported.")


def one_shot(model, t_sparsity):
    """Prune a model without refitting it.

    Args:
      model: keras model
      t_sparsity (float): The desired sparsity of the model after pruning
    Returns:
      pruned model
    """
    return condense.optimizer.model_operations.pruning.prune_model(model,
                                                                   t_sparsity,
                                                                   in_place=False)
