import keras
import pytest
import condense


@pytest.fixture
def example_model():
    """Prepare test keras model."""
    return keras.models.load_model('tests/test_model.h5')


def test_simple_one_shot(example_model):
    """Simple one shot pruning run."""
    assert condense.one_shot(example_model, 0.5)
