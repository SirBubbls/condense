"""This file contains the implementation for basic pruning operations on arbitrary torch modules."""
import torch
import torch.nn as nn
import copy
from condense.optimizer.sparsity_functions import Constant


def calc_parameter_sparsity(p):
    """Calculates the sparsity percentage of a torch parameter.

    Args:
      p: torch parameter
    Returns: sparsity percentage (as float) with range [0.0, 1.0]
    """
    x = torch.sum((torch.flatten(p) == 0).float())
    return float(x) / p.numel()


def masking_fn(X, t_sparsity):
    """Default masking function used by the PruningAgent."""
    threshold = torch.sort(torch.abs(X).flatten())[0][int(len(X.flatten()) * t_sparsity.get_epoch_sparsity())]
    return (torch.abs(X) > threshold).float()


class PruningAgent(nn.Module):
    """This class augments an existing torch module with callbacks and parameter masks."""

    def __init__(self, model, strategy):
        """You need to pass a module and a constant sparsity strategy.

        Args:
           model (torch.nn.Module): existing torch module
           strategy: the sparisty target strategy
        """
        super(PruningAgent, self).__init__()

        if not isinstance(strategy, Constant):
            raise Exception('Currently only the constant sparsity strategy is supported.')
        self.model = model

        # Parameter masks
        self.mask = {}
        self.layer_strategies = self.__init_per_layer_sparsity_strategies(strategy)
        self.masking_fn = masking_fn

        self.__init_parameter_masks()
        self.__wrap_sub_modules()

    def __init_per_layer_sparsity_strategies(self, strategy):
        strat = {}
        for p in self.model.parameters():
            strat[p] = copy.deepcopy(strategy)
            strat[p].set_base_sparsity(calc_parameter_sparsity(p))

        return strat

    def __init_parameter_masks(self, initialize_ones=False):
        """Initialize parameter masks.

        Args:
          initialize_ones (boolean): initialize mask values as 1 (no masking)
        """
        for p in self.model.parameters():
            if initialize_ones:
                self.mask[p] = torch.ones(p.size())
            else:
                self.mask[p] = self.masking_fn(p, self.layer_strategies[p])
                p.data = p.data * self.mask[p]  # apply mask to corresponding parameter

    def __wrap_sub_modules(self):
        """Applies pruning functionality to every parameter of the actual model."""
        for param in self.model.parameters():
            param.register_hook(lambda g, mask=self.mask[param]: g * mask)
            # param.register_hook(lambda g, p=param: self.__update_parameter_mask(p))
            # param.register_hook(lambda g, p=param: self.layer_strategies[p].next_epoch())

    def __update_parameter_mask(self, p):
        """Update masks for a parameter p."""
        self.mask[p] = self.masking_fn(p, self.layer_strategies[p])

    def get_parameter_sparsity(self):
        """Get a list of the sparsity percentages of every model parameter."""
        return [calc_parameter_sparsity(p) for p in self.model.parameters()]
