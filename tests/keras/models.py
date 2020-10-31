"""This module provides trained keras models for tests."""
import keras
import tensorflow_datasets as tfds
from keras.models import Sequential
from keras.layers import Dense

# Construct a tf.data.Dataset
iris_ds = tfds.load('iris', split='train', shuffle_files=True, as_supervised=True).repeat()

iris_model = Sequential([
    Dense(40, input_shape=(4,), activation='relu'),
    Dense(80, activation='relu'),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
iris_model.compile('adam', 'sparse_categorical_crossentropy')
iris_model.fit(iris_ds.batch(300), epochs=50, steps_per_epoch=5, verbose=0)


def iris():
    """Model trained on the iris dataset."""
    n = keras.models.clone_model(iris_model)
    n.compile('adam', 'sparse_categorical_crossentropy')
    return n
