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

    def __init__(self, model, strategy=None, apply_mask=False, ignored_params=[]):
        """You need to pass a module and a constant sparsity strategy.

        Args:
           model (torch.nn.Module): existing torch module
           strategy: the sparisty target strategy
           apply_mask (boolean): if this is true a mask will get generated and applied on initialization
           ignored_params (list): no pruning gets applied onto the element in the list
        """
        super(PruningAgent, self).__init__()

        self.model = model

        # create a list of parameters to prune
        _ignored_params = []
        for param in ignored_params:
            if isinstance(param, nn.Module):
                _ignored_params.extend(list(param.parameters()))
            elif isinstance(param, nn.parameter.Parameter):
                _ignored_params.append(param)
            else:
                raise Exception('only parameters and modules are supported in argument ignored_params')

        self.to_prune = self.__get_parameters_to_prune(_ignored_params)

        # Parameter masks
        self.mask = {}
        self.masking_fn = masking_fn
        if strategy:
            if not isinstance(strategy, Constant):
                raise Exception('Currently only the constant sparsity strategy is supported.')
            self.layer_strategies = self.__init_per_layer_sparsity_strategies(strategy)

        self.init_parameter_masks(not apply_mask)
        self.__wrap_sub_modules()

    def __get_parameters_to_prune(self, ignored_params):
        params = []

        for param in self.model.parameters():
            is_ignored = False
            for ignored_param in ignored_params:
                if param is ignored_param:
                    is_ignored = True
                    break

            if not is_ignored:
                params.append(param)

        return params

    def __init_per_layer_sparsity_strategies(self, strategy):
        strat = {}
        for p in self.model.parameters():
            strat[p] = copy.deepcopy(strategy)
            strat[p].set_base_sparsity(calc_parameter_sparsity(p))

        return strat

    def init_parameter_masks(self, initialize_ones=True):
        """Initialize parameter masks.

        Args:
          initialize_ones (boolean): initialize mask values as 1 (no masking)
        """
        for p in self.to_prune:
            if initialize_ones:
                self.mask[p] = torch.ones(p.size())
            else:
                self.mask[p] = self.masking_fn(p, self.layer_strategies[p])
                p.data = p.data * self.mask[p]  # apply mask to corresponding parameter

    def __wrap_sub_modules(self):
        """Applies pruning functionality to every parameter of the actual model."""
        for param in self.to_prune:
            param.register_hook(lambda g, p=param: g * self.mask[p])
            # param.register_hook(lambda g, p=param: self._update_parameter_mask(p))
            # param.register_hook(lambda g, p=param: self.layer_strategies[p].next_epoch())

    def _update_parameter_mask(self, p):
        """Update masks for a parameter p."""
        self.mask[p] = self.masking_fn(p, self.layer_strategies[p])

    def get_parameter_sparsity(self):
        """Get a list of the sparsity percentages of every model parameter."""
        return [calc_parameter_sparsity(p) for p in self.model.parameters()]
