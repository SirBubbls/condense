"""Tests imports of public functions"""
import condense


def test_keras_import():
    assert condense.keras


def test_keras_callback_import():
    assert condense.keras.callbacks
    assert condense.keras.callbacks.PruningCallback

def test_keras_wrapper_import():
    assert condense.keras.wrappers
    assert condense.keras.wrappers.PruningWrapper
