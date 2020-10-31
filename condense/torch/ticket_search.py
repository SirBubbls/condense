"""This module implements a use case for the lottery ticket hypothesis.

For a detailed explanation of this hypothesis check out https://arxiv.org/abs/1803.03635.
There is also an implementation available at https://github.com/google-research/lottery-ticket-hypothesis.
"""
import torch
import logging
from . import PruningAgent

logger = logging.getLogger('condense')


class TicketSearch():
    """This class can be used to search for a winning ticket of an arbitrary torch module."""

    def __init__(self, agent):
        """A torch module wrapped in a PruningAgent is required."""
        assert isinstance(agent, PruningAgent), 'agent has to be a PruningAgent'
        self.agent = agent
        self._original_params = []

    def __check_mask(self):
        for mask in self.agent.mask.values():
            if (mask.numpy() == 0).any():
                return False
        return True

    def _reinitialize_parameters(self):
        # initialize masks
        for mask in self.agent.mask.values():
            mask.data = torch.ones(mask.size())

        # TODO reinitialize all parameters

    def __enter__(self):
        """Stores all parameter configurations."""
        if not self.__check_mask():
            logger.warning('🚨 Parameters are already masked')
            self._reinitialize_parameters()
            assert self.__check_mask()

        logger.info('💾 Storing module parameters for reinitialization')

        # Saving parameters
        self._original_params = [p.clone().detach() for p in self.agent.model.parameters()]

        logger.info('🔬 Searching for winning ticket')

    def __exit__(self, type, value, traceback):
        """Reinitializes all parameters and applies masks."""
        logger.info('🎟 Winning ticket found')

        logger.info('⚙️ Generating Mask')
        self.agent.init_parameter_masks(initialize_ones=False)

        # reinitialized weights & and apply mask to reinitialized tensors
        for p, w in zip(self.agent.model.parameters(), self._original_params):
            p.data = w.data * self.agent.mask[p]

        logger.info('😄 Reinitialized module parameters')
        logger.info('🥷 Ticket masks applied to module parameters')
