# About the project
    
This projects is the result of my Bachelor Thesis ['Implementierung und Vergleich von Pruning Methoden für künstliche neuronale Netze'](https://www.github.com/sirbubbls/pruning-ba) ([PDF]()).
It provides easy tools to prune Keras and Torch models through easy python interfaces. 
Check out [Quick Start](/quick_start.md) to get started as fast as possible.
You can find practical examples in form of Jupyter notebooks [here](https://github.com/SirBubbls/condense/tree/master/notebook).

The project is open source and licensed under MIT so contributions and further developments are welcome.


# Features
Condense currently provides compatibility modules for [PyTorch](https://pytorch.org) and [Keras/TensorFlow](https://keras.io).
API's for both modules are currently not unified, so there is a difference which backend you use.

**Support for PyTorch**  
Because of a more generic pruning implementation all `torch` layers are supported on a basic level.
Pruning targets are all `torch` model parameters, so no specific pruning operations like filter pruning or unit pruning are currently implemented.

**Support for Keras/TensorFlow**  
Because a completely layer agnostic implementation is currently not possible.
Following Layers are currently fully supported:
- Dense
- Conv2D

Many other layers similar layer types like Conv1D or Conv3D are supported.
But you may run into problems using completely different layer architectures like LSTMs or BatchNorm.
By default not supported layers won't get pruned and a warning gets displayed.

