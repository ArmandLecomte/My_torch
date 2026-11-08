# MyTorch

A reimplementation of PyTorch, with fewer features.

It also features a chessboard analyzer made with it and a 

## Requirements

- Python 3.12+

> [!TIP]
> We suggest that you use a virtual environment for python (e.g., [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

### Installation of the library
```bash
pip install ./my_torch
```

### Build and run the command-line tools

```bash
python -m pip install -r requierements.txt
make
./my_torch_analyzer --help
./my_torch_generator --help
```

`make` installs the project dependencies using `python3 -m pip` before building
the command-line tools. Activate your virtual environment first if you want the
dependencies installed there.

Build one tool with `make analyser` or `make generator`. The generator takes
pairs of configuration file and model count, and writes the generated models
next to each configuration file:

```bash
./my_torch_generator config.json 3
```

The library API for generating a network from a JSON configuration is
`my_torch.generate_neural_network`. Each call reads the configuration and
creates a fresh network with independently sampled weights and biases. For
example, four networks can be generated from the same file like this:

```python
from my_torch import generate_neural_network

networks = [generate_neural_network("config1.json") for _ in range(4)]
```

A configuration specifies the input size, hidden layers, output layer, and an
optional learning-rate scheduler. Each layer requires a neuron count, an
activation name, and weight and bias initializers. Supported activations are
`Sigmoid`, `Heaviside`, `ReLU`, `LeakyReLU`, `Linear`, `Tanh`, `Softplus`, and
`Softmax`; initializer types are `uniform`, `normal`, and `constant`.

```json
{
  "input_count": 3,
  "hidden_layers": [
    {
      "neurons": 8,
      "activation_function": "ReLU",
      "weight_initializer": {"type": "uniform", "low": -0.1, "high": 0.1},
      "bias_initializer": {"type": "constant", "value": 0.0}
    }
  ],
  "output_layer": {
    "neurons": 2,
    "activation_function": "Softmax",
    "weight_initializer": {"type": "normal", "mean": 0.0, "std": 0.05},
    "bias_initializer": {"type": "constant", "value": 0.0}
  },
  "lr_scheduler": {"type": "exponential", "initial_lr": 0.1, "decay_rate": 0.98}
}
```

Scheduler types are `constant`, `exponential`, and `step`; the `step` scheduler
accepts an optional `gamma` (default `0.1`). If `lr_scheduler` is omitted, the
library's default constant learning rate is used. The loss defaults to MSE.
`Heaviside` has no backward implementation, so it cannot be used in layers that
need gradient-based training.

### Run the Jupyter benchmark

The notebook `my_torch/notebooks/neural_network_benchmark.ipynb` trains and compares
three neural-network configurations on the FEN dataset in `my_torch/dataset`. From
the repository root, install the library and its benchmark dependencies, plus the
FEN parser used by the notebook:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "my_torch[benchmark]"
python -m pip install -e .
```

Launch Jupyter from the notebook directory so its relative dataset path resolves
correctly:

```bash
jupyter lab my_torch/notebooks/neural_network_benchmark.ipynb
```
