# Quick Start
## Installation
You can install this module with `pip` through git for the most up to date features or you can just use the PyPi Repository. 

- **PyPi**: `pip install condense` (https://pypi.org/project/condense/)
- **Git Repository**: `python -m pip install git+https://github.com/SirBubbls/condense.git`

## Usage 

### One-Shot Prune a Model

The `condense.one_shot` interface is the easiest way to prune a keras model.
More information about possible function arguments can be found [here](/pdoc/condense/index.html).

> **Caution!**   
> One-Shot pruning is not recommended way of pruning and should therefore only used for experimenting.

```python
import keras
import condense 

model = keras.load_model('...')
pruned_model = condense.one_shot(model, 0.3)   # 30% desired model sparsity
```

### Keras/TensorFlow 2.0
A more sophisticated way of pruning can be utilized by using the `condense.keras` comparability module.

```python
import keras
import condense

model = keras.load_model('...')
augmented_model = condense.keras.prune_model(model)

# No pruning happened yet.
# First we need to fit the model with the a callback class assigned.
augmented_model.fit(gen, callbacks=[condense.keras.callbacks.PruningCallback()])
```

> For a real word example you can check out this 
[Convolutional Layer Pruning (MNIST).ipynb](https://github.com/SirBubbls/condense/blob/dev/notebook/Convolutional%20Layer%20Pruning%20(MNIST).ipynb )
Jupyter Notebook.


### PyTorch
Alternatively you can also use PyTorch models/modules with `condense`.
Actually using the PyTorch module provides way more flexibility for the user, than the keras equivalent.
It's probably a good idea to set the logging level to `info` (`condense.logger.setLevel('INFO')`), this will provide you with useful information during training/pruning.

For the most basic form of pruning you need to wrap your module with provided `condense.torch.PruningAgent` class.
```python
module = # your own torch module

pruned_module = condense.torch.PruningAgent(module,
                                            Constant(0.85),
                                            apply_mask=True,
                                            ignored_params=[module.<<output_name>>]  # optional but highly recommended
                                            )

# You can train your module just as you normally would 
module.train(...)
```
Now all masks were generated to achieve the $85\%$ layer sparsity target.
Your original module will now consider the sparsity masks during training.

The example below implements the lottery ticket method.
```python
from condense.torch import PruningAgent, TicketSearch
from condense.optimizer.sparsity_functions import Constant
module = ...

agent = PruningAgent(module, 
                     Constant(0.8),
                     apply_mask=False,    # required, because we don't want to mask our parameters yet
                     ignored_params=[module.output]
                     )

with TicketSearch(agent):
    # Do your training/searching here
    agent.model.train(...)  # ticket search training

# Weights are now reinitialized and sparsity mask is applied

agent.model.train(...)  # final training
```

> For more examples please have a look at some Jupyter Notebooks (https://github.com/sirbubbls/condense/tree/master/notebook)
