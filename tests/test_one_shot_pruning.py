import keras
import pytest
import condense
import sys
sys.path.append('tests/keras')
from models import iris

@pytest.fixture
def example_model():
    """Prepare test keras model."""
    return iris()


def test_simple_one_shot(example_model):
    """Simple one shot pruning run."""
    assert condense.one_shot(example_model, 0.5)
