"""Tests imports of public functions"""
import condense


def test_keras_import():
    """Testing condense.keras api import."""
    assert condense.keras
    assert condense.keras.wrappers
    assert condense.keras.wrappers.PruningWrapper
    assert condense.keras.callbacks
    assert condense.keras.callbacks.PruningCallback
