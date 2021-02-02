"""Tests for model utility functions."""
import keras
import sys
from condense.utils.model_utils import calc_model_sparsity
sys.path.append('tests/keras')
from models import iris

def test_calc_model_sparsity():
    """Testing model sparsity calculation."""
    model = iris()
    assert calc_model_sparsity(model)
