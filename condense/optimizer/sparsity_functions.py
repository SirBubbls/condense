"""This module contains sparsity function implementations."""
import abc
import numpy as np


class SparsityFunction(metaclass=abc.ABCMeta):
    """Base class for sparsity functions."""
    def __init__(self):
        """."""
        self.__epoch = 0
        self.params = None
        self.__base_sparsity = None

    @abc.abstractmethod
    def get_epoch_sparsity(self):
        """Get the current target sparsity for this epoch."""

    def set_training_params(self, params):
        """Set training parameters like epochs or steps_per_epoch for this training instance.

        This function should not be overwritten.
        """
        self.params = params

    def next_epoch(self):
        """Call this function on new epoch."""
        self.__epoch += 1

    def get_epoch(self):
        """Get the current epoch.

        Returns:
          current epoch
        """
        return self.__epoch

    def set_base_sparsity(self, val):
        """Set base layer sparsity.

        Args:
          val: sparsity
        """
        if not isinstance(val, float):
            raise ValueError(f'val should be a float not {type(val)}')
        self.__base_sparsity = val

    def get_base_sparsity(self):
        """Get configured base sparsity of this sparsity function.

        Returns:
          base sparsity
        """
        if self.__base_sparsity is None:
            raise Exception("""__base_sparsity is not yet known.
            Use set_base_sparsity before calling set_training_params.""")
        return self.__base_sparsity


class Constant(SparsityFunction):
    """Have a constant target layer sparsity."""
    def __init__(self, sparsity):
        """Constant sparsity function constructor.

        Args:
          sparsity: the desired (constant) target sparsity
        """
        super(Constant, self).__init__()
        self.__target = sparsity

    def get_epoch_sparsity(self):
        """Get the constant sparsity value.

        Returns:
          target sparsity for current epoch.
        """
        return self.__target


class Linear(SparsityFunction):
    """A linear increase of sparsity over the duration of the training operation."""
    def __init__(self, target_sparsity):
        """.

        Args:
          target_sparsity: Sparsity the layer should have after finished training.
        """
        super(Linear, self).__init__()
        self.__target = target_sparsity
        self.strategy = None

    def set_training_params(self, params):
        """Set training parameters like epochs or steps_per_epoch for this training instance."""
        super(Linear, self).set_training_params(params)

        if 'epochs' not in self.params:
            raise Exception('Number of epochs is not known, but required for calculating strategy.')

        self.strategy = np.linspace(
            self.get_base_sparsity(),
            self.__target,
            self.params['epochs']
        )

    def get_epoch_sparsity(self):
        """Get the current target sparsity for this epoch."""
        if self.strategy is None:
            raise Exception('No strategy pre-calculated yet.')

        return self.strategy[self.get_epoch()]
