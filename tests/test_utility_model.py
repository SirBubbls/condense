"""Tests for model utility functions."""
import keras
from condense.utils.model_utils import calc_model_sparsity


def test_calc_model_sparsity():
    """Testing model sparsity calculation."""
    model = keras.models.load_model('tests/test_model.h5')
    assert calc_model_sparsity(model)
