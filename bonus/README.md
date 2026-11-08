# Neural network viewer

From the project root, install the project dependencies and launch the viewer:

```bash
python -m pip install -r requierements.txt
python bonus/neural_network_viewer.py
```

Select a `.nn` file on the left, use **Refresh model list** to find newly added or
removed models, and click a neuron to inspect its bias and strongest weights.
Large layers are abbreviated in the diagram; the neuron and weight counts remain
visible.

## Numba CUDA trainer

Install the optional CUDA dependency:

```bash
python -m pip install -r bonus/requirements.txt
```

`NumbaGPUTrainer` trains a `my_torch.NeuralNetwork` on an NVIDIA GPU with
mini-batch gradient descent. The machine needs a supported NVIDIA GPU and CUDA
driver; the class raises a clear error if CUDA is unavailable. It currently
supports the MSE loss and the `Sigmoid`, `ReLU`, `LeakyReLU`, `Linear`, `Tanh`,
`Softplus`, and `Softmax` activations. `Heaviside` and custom losses are not
supported.

```python
from bonus.numba_gpu_trainer import NumbaGPUTrainer
from my_torch import NeuralNetwork

network = NeuralNetwork.load("my_torch/trained_models/baseline.nn")
trainer = NumbaGPUTrainer(batch_size=32)
accuracy, losses = trainer.train(network, train_data, num_epoch=100)
```

The trainer updates the supplied network and its learning-rate scheduler in
place. Unlike `NeuralNetwork.train`, it performs one averaged gradient update
per mini-batch rather than one update per sample.

The `my_torch_analyser --train` command automatically selects this trainer when
Numba CUDA is available and the network uses a supported loss and activations.
Otherwise, it reports the reason and uses the regular CPU trainer.
