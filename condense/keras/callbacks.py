"""This module implements a keras callback object used for the model.fit() operation.

Use the Callback like:
```python
model.fit(x, y, callbacks=[PruningCallback()])
```
"""
from keras.callbacks import Callback
from condense.keras.wrappers import PruningWrapper


class PruningCallback(Callback):
    """Required for .fit operation."""
    def on_epoch_end(self, epoch, logs=None):
        """Prunes every layer in the model, that is Wrapped by a PruningWrapper."""
        for layer in self.model.layers:
            if isinstance(layer, PruningWrapper):
                # Pruning Operation
                layer.prune(0.35)
