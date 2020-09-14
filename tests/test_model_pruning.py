"""Model pruning module tests."""
import keras
import pytest
from logging import info
from condense.optimizer.model_operations.pruning import prune_model
from condense.utils.model_utils import calc_model_sparsity


@pytest.fixture
def example_model():
    """Prepare test keras model."""
    return keras.models.load_model('tests/test_model.h5')


def test_simple_model_pruning(example_model):
    """Check simple pruned model."""
    original_weights = example_model.get_weights()
    new_model = prune_model(example_model, 0.2)
    assert (new_model.get_weights()[0] != original_weights[0]).any(), 'weights should change'


def test_example_model(example_model):
    """Checking if model requirements are met for the test model."""
    info(f'Base Model Sparsity: {calc_model_sparsity(example_model)}')
    assert calc_model_sparsity(example_model) < 0.1, 'base sparsity of the model should be under 10%'


def test_model_pruning_sparsity(example_model, t_sparsity=0.5, accepted_delta=0.03):
    """Test sparsity change after pruning."""
    unpruned_sparsity = calc_model_sparsity(example_model)
    prune_model(example_model, t_sparsity)
    pruned_sparsity = calc_model_sparsity(example_model)
    assert abs(unpruned_sparsity - t_sparsity) > 0.4, 'base model is to sparse'
    assert abs(t_sparsity - pruned_sparsity) < accepted_delta, 'missed target delta'


def test_in_place(example_model):
    """This test checks if in_place parameter works as expected."""
    o_weights = example_model.get_weights()
    n_model = prune_model(example_model, 0.8, in_place=False)
    assert (example_model.get_weights()[0] == o_weights[0]).all(), \
        'original model shouldn\'t change'
    assert (n_model.get_weights()[0] != o_weights[0]).any(), 'return pruned model'
