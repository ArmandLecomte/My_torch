import numpy as np

from my_torch import NeuralNetwork, LayersDefinition

layers = LayersDefinition(
    input_size=30,
    hidden_layers=[5],
    output_size=3,
    default_weight=lambda: float(np.random.randn()),
    default_bias=lambda: float(np.random.randn())
)
nn = NeuralNetwork(layers)


if __name__ == "__main__":
    out = nn.forward(np.array([0, 0, 1, 1, 0, 0,
       0, 1, 0, 0, 1, 0,
       1, 1, 1, 1, 1, 1,
       1, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 1]).reshape(1, 30))

    print(out)
