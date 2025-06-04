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
                # Forward pass
                A = X_batch
                for layer in self.layers:
                    A = layer.forward(A)

                # Loss
                loss = self.loss.loss(A, y_batch)
                current_epoch_loss += loss
                num_batches += 1

                # Backward pass with immediate weight updates
                dA = self.loss.loss_derivative(A, y_batch)
                for layer in reversed(self.layers):
                    dA, dW, db = layer.backward(dA)
                    self.optimizer.step(layer, dW, db)

            avg_epoch_loss = current_epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

        return epoch_losses


    def predict(self, X_test):
        A = X_test
        for layer in self.layers:
            A = layer.forward(A)
        return A
