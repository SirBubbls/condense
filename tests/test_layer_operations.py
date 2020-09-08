import numpy as np
import condense.optimizer.layer_operations as op


def test_u_prune_layer():
    """Unit/Neuron Pruning Test Function"""
    t_ndarray = np.array([[1, 2], [0.2, 0.4], [0, 2]])
    pruned_ndarray = op.unit_prune.u_prune_layer(t_ndarray, 0.8)
    assert not pruned_ndarray.sum(0)[0]
    assert (t_ndarray != pruned_ndarray).any(), 'base array shouldn\'t change'
    assert op.unit_prune.u_prune_layer(t_ndarray).any(), 'target sparsity should be optional'
