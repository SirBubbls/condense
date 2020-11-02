import torch
import pytest
import condense
import numpy as np
import tensorflow_datasets as tfds
import torch.nn as nn
from logging import info


ds_train, ds_test = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

def generator(batch_size, data_set):
    _gen = iter(tfds.as_numpy(data_set.batch(batch_size).cache()))
    while True:
        X, y = next(_gen)
        yield torch.Tensor(X.reshape(batch_size, 1, 28, 28)), torch.Tensor(y).type(torch.LongTensor)

gen = generator(300, ds_train)


@pytest.fixture
def conv_model():
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
            self.layer1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2)
            self.layer2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=2)
            self.dense = nn.Linear(288, out_features=50)
            self.output = nn.Linear(50, out_features=10)

        def forward(self, X):
            X = self.layer1.forward(X)
            X = self.layer2.forward(torch.relu(X))
            X = X.view(torch.relu(X).size(0), -1)
            X = self.dense.forward(torch.relu(X))
            X = self.output.forward(torch.relu(X))
            X = torch.log_softmax(X, 1)
            return X

        def train(self, gen, epochs=20):
            criterion = nn.CrossEntropyLoss()
            optim = torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=0.1)

            for _ in range(epochs):
                X, y = next(gen)
                for _ in range(20):
                    self.zero_grad()
                    pred = self.forward(X)
                    l = criterion(pred, y)
                    l.backward()
                    optim.step()
                print('Training Loss:', l)

    return Network()


def test_reinitialization(conv_model):
    """Checks if the parameters get reinitialized to the original parameters."""
    pruned = condense.torch.PruningAgent(conv_model, condense.optimizer.sparsity_functions.Constant(0.7), apply_mask=False)
    pre_pruning = [p.clone().detach().numpy() for p in pruned.model.parameters()]

    # search
    with condense.torch.TicketSearch(pruned):
        masks = pruned.model.train(gen, 10)

    after_pruning = [p.clone().detach().numpy() for p in pruned.model.parameters()]
    ticket_mask = [m.detach().numpy() for m in pruned.mask.values()]

    # reinit tests
    for pre, after, mask in zip(pre_pruning, after_pruning, pruned.mask.values()):
        assert (pre * mask.numpy() == after).all(), 'lottery ticket search params were not reinitialized correctly'

    # training on model
    pruned.model.train(gen, 20)

    # check if mask changed
    for old, p in zip(ticket_mask, pruned.model.parameters()):
        assert (old == pruned.mask[p].numpy()).all(), 'mask changed during training'

    # mask was considered during training
    for param in pruned.to_prune:
        assert ((param.detach().numpy() != 0) == pruned.mask[param].detach().numpy()).all()
