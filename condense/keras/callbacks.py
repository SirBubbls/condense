"""This module implements a keras callback object used for the model.fit() operation.

Use the Callback like:
```python
model.fit(x, y, callbacks=[PruningCallback()])
```
"""
from keras.callbacks import Callback
from condense.keras.wrappers import PruningWrapper
from condense.utils.layer_utils import calc_layer_sparsity


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
