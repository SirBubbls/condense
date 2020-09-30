"""This module implements a keras layer Wrapper."""
import keras
import tensorflow as tf


class PruningWrapper(keras.layers.Wrapper):
    """Wrapper for Keras Dense Layers.

    Todo:
        * Make wrapper usable on all kinds of layers not only Dense.
    """
    def __init__(self, layer):
        """Override init.

        parent:
        """
        super(PruningWrapper, self).__init__(layer)
        self.units = 10

    def prune(self, target_sparsity):
        """Pruning operation on layer."""
        # Calc Threshold
        abs_weights = tf.sort(tf.reshape(tf.math.abs(self.w), [-1]))

        size = tf.cast(tf.shape(abs_weights)[0], dtype=tf.float32)
        threshold = tf.gather(abs_weights,
                              tf.cast(size * target_sparsity, dtype=tf.int32))
        # tf.print(tf.cast(tf.cast(tf.shape(abs_weights)[0], dtype=tf.float32) * .2, dtype=tf.int32))
        # tf.print(threshold)
        mask = tf.cast(tf.math.greater_equal(self.w, threshold), dtype=tf.float32)

        # Apply mask on weight layer
        self.w.assign(self.w * mask)

        # Update Step
        self.step.assign(self.step + 1)
        return tf.no_op('Pruning')

    def build(self, input_shape):
        """Override build method.

        parent:
        """
        self.step = tf.Variable(0, trainable=False, dtype=tf.int32, name='step')
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.units,),
            initializer='zero',
            trainable=True
        )
        # self.mask = self.add_weight(
        #     name='sparsity_mask',
        #     shape=(input_shape[-1], self.units),
        #     initializer='ones',
        #     trainable=False
        # )

    def call(self, inputs, training=False):
        """Override call method.

        parent:
        """
        return tf.matmul(inputs, self.w) + self.b
