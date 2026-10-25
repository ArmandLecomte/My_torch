import numpy as np

from my_torch import NeuralNetwork, LayerSettings, sigmoid, sigmoid_derivative, \
    NNSettings, mse_loss, mse_derivative

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


if __name__ == "__main__":
    out = nn.forward(np.array([0, 0, 1, 1, 0, 0,
       0, 1, 0, 0, 1, 0,
       1, 1, 1, 1, 1, 1,
       1, 0, 0, 0, 0, 1,
       1, 0, 0, 0, 0, 1]).reshape(1, 30))

    num_epochs = 1000
    for epoch in range(num_epochs):
        epoch_losses = []

        for x_sample, y_sample in zip(x_train, y_train):
            # back_propagation updates weights and returns the loss for this step
            loss = nn.back_propagation(x_sample, y_sample, learning_rate=0.01)
            epoch_losses.append(loss)

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.6f}")

    # Accessing the loss history stored inside the network instance:
    print(f"Final loss recorded in network: {nn.last_loss}")
    print(out)
