"""Supported Keras Layers are defined in this module."""
from keras import layers

# List of currently supported Keras Layers
SUPPORTED_LAYERS = [
    layers.Dense,
    layers.Conv2D
]


def is_supported_layer(layer):
    """This function tests if a layer is officially supported by this framework.

    Args:
      layer: Layer to test
    Returns:
      True if supported. False if not supported.
    """
    for supported_layer in SUPPORTED_LAYERS:
        if isinstance(layer, supported_layer):
            return True
    return False
