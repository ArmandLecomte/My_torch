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

Install the optional Numba CUDA package and CUDA runtime/compiler libraries:

```bash
python -m pip install -r bonus/requirements.txt
```

`NumbaGPUTrainer` trains a `my_torch.NeuralNetwork` on an NVIDIA GPU with
shuffled mini-batch gradient descent. The machine needs a supported NVIDIA GPU
and CUDA driver; the requirements install CUDA 12 runtime and compiler libraries
in the Python environment. It supports MSE, or categorical cross-entropy with a
Softmax output, and the `Sigmoid`, `ReLU`, `LeakyReLU`, `Linear`, `Tanh`,
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

CUDA work is distributed by sample for forward, activation, and delta kernels,
and by parameter with a parallel batch reduction for updates. This gives
mini-batches more independent GPU blocks than the previous flat launch. The
Numba low-occupancy advisory is ignored only for parameter-update kernels with
fewer than 128 parameters, where the model itself limits the available blocks;
other low-occupancy warnings remain visible.

The benchmark notebook uses Softmax with categorical cross-entropy for GPU
training, shuffles samples every epoch, and uses a higher learning rate than
the CPU reference. Its plots still report MSE and classification accuracy.

The command-line analyzer uses its regular trainer by default. To explicitly
request the bonus GPU trainer, install the optional dependency above and pass
`--bonus`:

```bash
./my_torch_analyser --train --bonus \
  my_torch/trained_models/baseline.nn \
  my_torch/dataset/checkmate/many_pieces.txt
```

Without `--bonus`, the optional trainer is not loaded. If `--bonus` is requested
but the trainer is unavailable, training falls back to the standard CPU trainer.
Networks unsupported by the bonus trainer are rejected.
