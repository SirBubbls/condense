"""This module implements a keras layer Wrapper."""
import keras
import tensorflow as tf
from .support import is_supported_layer


class PruningWrapper(keras.layers.Wrapper):
    """Wrapper for Keras Dense Layers."""
    def __init__(self, layer, strategy):
        """Override init.

        parent:
        """
        if not is_supported_layer(layer):
            raise ValueError(f'Layer {layer.name} is not supported')

        super(PruningWrapper, self).__init__(layer,
                                             name=f'pruned_{layer.name}')
        self.layer = layer
        self.strategy = strategy
        self.mask = self.add_weight(
            name=f'{layer.name}_mask'
        )
        if isinstance(layer, keras.layers.Dense):
            self.units = self.layer.units

    @property
    def kernel(self):
        """Returns the pruned kernel of this layer."""
        return self.layer.kernel * self.mask

    def prune(self, t_sparsity=None):
        """Execute pruning operation on layer.

        Args:
          t_sparsity (float): if you want to run the pruning manually
                              you can set a desired sparsity through this argument
        """
        if not t_sparsity:
            t_sparsity = self.strategy.get_epoch_sparsity()

        # Calc Threshold
        abs_weights = tf.sort(tf.reshape(tf.math.abs(self.layer.kernel), [-1]))

        size = tf.cast(tf.shape(abs_weights)[0], dtype=tf.float32)
        threshold = tf.gather(abs_weights,
                              tf.cast(size * t_sparsity, dtype=tf.int32))

        # Assign Mask
        self.mask.assign(tf.cast(tf.math.greater_equal(tf.math.abs(self.layer.kernel), threshold), dtype=tf.float32))

        # Apply mask on weight layer
        self.layer.kernel.assign(self.layer.kernel * self.mask)

        # Update Step
        return tf.no_op('Pruning')

    def build(self, input_shape):
        """Override build method.

        parent:
        """
        super(PruningWrapper, self).build(input_shape)
        self.layer.build(input_shape)
        self.layer.built = True
        self.__build_mask(input_shape)
        self.built = True

    def __build_mask(self, input_shape):
        """Initialzie mask for each correspondig layer.

        Todos:
            * make masks based on layer.get_weights() -> layer type agnostic
        """
        # Dense Layer
        if isinstance(self.layer, keras.layers.Dense):
            self.mask = self.add_weight(
                name=f'{self.layer.name}_mask',
                shape=(input_shape[-1], self.units),
                initializer='ones',
                trainable=False
            )


    def call(self, inputs, training=False):
        """Override call method.

        parent:
        """
        self.layer.kernel.assign(self.layer.kernel * self.mask)
        return self.layer.call(inputs)
