"""This module implements a keras callback object used for the model.fit() operation.

Use the Callback like:
```python
model.fit(x, y, callbacks=[PruningCallback()])
```
"""
import tensorflow as tf
from numpy import average, prod, array
from keras.callbacks import Callback
from condense.keras.wrappers import PruningWrapper
from condense.utils.layer_utils import calc_layer_sparsity


SUPPORTED_SCALARS = ['sparsity']


class PruningCallback(Callback):
    """This class is required as a callback for the keras.fit function."""
    def on_train_begin(self, logs=None):
        """This function is responsible for setting up the pruning strategies for each individual layer."""
        if not self.params['epochs']:
            raise Exception('Number of epochs not known so no pruning strategy can be calculated.')

        for layer in self.model.layers:
            if isinstance(layer, PruningWrapper):
                layer.strategy.set_base_sparsity(
                    calc_layer_sparsity(
                        layer.layer.kernel.numpy()
                    )
                )
                layer.strategy.set_training_params(self.params)

    def on_epoch_end(self, epoch, logs=None):
        """Prunes every layer in the model, that is Wrapped by a PruningWrapper."""
        for layer in self.model.layers:
            if isinstance(layer, PruningWrapper):
                # Pruning Operation
                layer.prune()

        for layer in self.model.layers:
            if isinstance(layer, PruningWrapper):
                layer.strategy.next_epoch()


class Tensorboard(Callback):
    """You can use this tensorflow callback to log pruning specific metrics to tensorboard."""
    def __init__(self, scalars: list):
        """Constructor.

        Args:
          scalars (list): a list of scalars to track
        """
        super(Tensorboard, self).__init__()

        for scalar in scalars:
            if scalar not in SUPPORTED_SCALARS:
                raise ValueError(f'Scalar {scalar} is not supported.')

        self.scalars = scalars

    def _calc_model_sparsity(self):
        sparsity = []
        weights = []

        for layer in self.model.layers:
            if isinstance(layer, PruningWrapper):
                sparsity.append(calc_layer_sparsity(layer.layer.kernel.numpy()))
                weights.append(prod(layer.layer.kernel.shape))
            elif hasattr(layer, 'kernel'):
                sparsity.append(calc_layer_sparsity(layer.kernel.numpy()))
                weights.append(prod(layer.kernel.shape))

        sparsity, weights = array(sparsity), array(weights)
        return float(average(sparsity, weights=weights))

    def on_epoch_end(self, epoch, logs=None):
        """This function runs every epoch end."""
        if 'sparsity' in self.scalars:
            tf.summary.scalar('model sparsity', data=self._calc_model_sparsity(), step=epoch)
