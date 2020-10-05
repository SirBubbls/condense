# Quick Start
## Installation
You can install this module with ‘pip’ through git for the most up to date features or you can just use the PyPi Repository. 

- **PyPi**: `pip install condense` (https://pypi.org/project/condense/)
- **Git Repository**: `python -m pip install git+https://github.com/SirBubbls/condense.git`

## Usage 

### One-Shot Prune a Model

The `condense.one_shot` interface is the easiest way to prune a keras model.

> **Caution!**   
> One-Shot pruning is not recommended way of pruning and should therefore only used for experimenting.

```python
import keras
import condense 

model = keras.load_model('...')
pruned_model = condense.one_shot(model, 0.3)   # 30% desired model sparsity
```

### Keras/TensorFlow 2.0
A more sophisticated method of pruning is using the `condense.keras` comparability module.

```python
import keras
import condense

model = keras.load_model('...')
augmented_model = condense.keras.prune_model(model)

# No pruning happened yet.
# First we need to fit the model with the a callback class assigned.
augmented_model.fit(gen, callbacks=[condense.keras.callbacks.PruningCallback()])
```
