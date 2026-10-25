import numpy as np
from matplotlib import pyplot as plt

from my_torch import NeuralNetwork, LayerSettings, sigmoid, sigmoid_derivative, NNSettings, mse_loss, mse_derivative

layersSettings = [
    LayerSettings(
        neurons=5,
        default_weight=lambda: float(np.random.randn()),
        default_bias=lambda: float(np.random.randn()),
        activation_function=sigmoid,
        derivative_activation=sigmoid_derivative
    ),
]

settings = NNSettings(
    input_count=30,
    hidden_layers=layersSettings,
    output_layer=LayerSettings(
        neurons=3,
        default_weight=lambda: float(np.random.randn()),
        default_bias=lambda: float(np.random.randn()),
        activation_function=sigmoid,
        derivative_activation=sigmoid_derivative
    ),
    loss_function=mse_loss,
    loss_derivative=mse_derivative,
)
nn = NeuralNetwork(settings)

# A
a =[0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1]
# B
b =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0]
# C
c =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0]

# Creating labels
y =[[1, 0, 0],
   [0, 1, 0],
   [0, 0, 1]]

x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30),
                                np.array(c).reshape(1, 30)]
y = np.array(y)

if __name__ == "__main__":
    data = [
        (x[0], y[0]),
        (x[1], y[1]),
        (x[2], y[2])
    ]

    acc, loss = nn.train(data, num_epoch=1000, learning_rate=0.01)

    plt.plot(acc)
    plt.ylabel('Accuracy')
    plt.xlabel("Epochs:")
    plt.show()

    # plotting Loss
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel("Epochs:")
    plt.show()


    def predict(x):
        Out = nn.forward(x)
        maxm = 0
        k = 0
        for i in range(len(Out[0])):
            if (maxm < Out[0][i]):
                maxm = Out[0][i]
                k = i
        if k == 0:
            print("Image is of letter A.")
        elif k == 1:
            print("Image is of letter B.")
        else:
            print("Image is of letter C.")
        plt.imshow(x.reshape(5, 6))
        plt.show()


    # Example: Predicting for letter 'B'
    for i in range(3):
        predict(x[i])
    print()
