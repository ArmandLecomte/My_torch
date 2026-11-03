# MyTorch

A reimplementation of PyTorch, with fewer features.

It also features a chessboard analyzer made with it and a 

## Requirements

- Python 3.15+

> [!TIP]
> We suggest that you use a virtual environment for python (e.g., [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

### Installation of the library
```bash
pip install ./my_torch
```

### Run the analyzer

```bash
pip install -r requirements.txt
python src/torch_analyzer/main.py --help
```

### Run the generator

```bash
pip install -r requirements.txt
python src/torch_generator/main.py --help
```

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
