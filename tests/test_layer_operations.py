import numpy as np
import condense.optimizer.layer_operations as op


def test_u_prune_layer():
    """Unit/Neuron Pruning Test Function"""
    t_ndarray = np.array([[1, 2], [0.2, 0.4], [0, 2]])
    pruned_ndarray = op.unit_prune.u_prune_layer(t_ndarray, 0.8)
    assert not pruned_ndarray.sum(0)[0]
    assert (t_ndarray != pruned_ndarray).any(), 'base array shouldn\'t change'
    assert op.unit_prune.u_prune_layer(t_ndarray).any(), 'target sparsity should be optional'


def test_w_prune_layer():
    """Weight Pruning Test Function"""
    t_ndarray = np.array([[1, 5, 3], [0, 2, 4]])
    pruned_ndarray = op.weight_prune.w_prune_layer(t_ndarray, 0.5)
    assert (t_ndarray != pruned_ndarray).any(), 'base array shouldn\'t change'
    assert (pruned_ndarray == np.array([[0, 5, 3], [0, 0, 4]])).all(), 'correct mask values'
    assert op.unit_prune.u_prune_layer(t_ndarray).any(), 'target sparsity should be optional'
