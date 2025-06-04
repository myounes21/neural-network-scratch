import numpy as np

from models.net_layer import Layer
from optimizers.mini_batch_sgd import MiniBatchSGD
from losses.cross_entropy import BinaryCrossEntropyLoss


class NN:
    def __init__(self, X, y, layer_sizes, activations):
        self.X = X
        self.y = y
        self.layers = []

        if len(layer_sizes) != len(activations):
            raise ValueError("The number of layer sizes must match the number of activation functions.")

        # Create layers with proper initialization
        input_dim = X.shape[0]
        for i, n_units in enumerate(layer_sizes):
            activation_type = activations[i]
            layer = Layer(n_units=n_units, activation=activation_type)
            layer.initialize(input_dim)
            self.layers.append(layer)
            input_dim = n_units

        self.loss = BinaryCrossEntropyLoss()
        self.optimizer = MiniBatchSGD()

    def train(self, epochs, learning_rate):
        self.optimizer.lr = learning_rate
        epoch_losses = []

        for epoch in range(epochs):
            current_epoch_loss = 0
            num_batches = 0

            for X_batch, y_batch in self.optimizer.get_batches(self.X, self.y):
                A = X_batch
                for layer in self.layers:
                    A = layer.forward(A)
                y_pred_batch = A

                loss = self.loss.loss(y_pred_batch, y_batch)
                current_epoch_loss += loss
                num_batches += 1

                dA = self.loss.loss_derivative(y_pred_batch, y_batch)
                batch_gradients_W = []
                batch_gradients_b = []
                for layer in reversed(self.layers):
                    dA_prev, dW, db = layer.backward(dA)
                    dA = dA_prev
                    batch_gradients_W.append(dW)
                    batch_gradients_b.append(db)

                batch_gradients_W.reverse()
                batch_gradients_b.reverse()

                for i, layer in enumerate(self.layers):
                    self.optimizer.step(layer, batch_gradients_W[i], batch_gradients_b[i])

            avg_epoch_loss = current_epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

        return epoch_losses

    def predict(self, X_test):
        A = X_test
        for layer in self.layers:
            A = layer.forward(A)
        return A
