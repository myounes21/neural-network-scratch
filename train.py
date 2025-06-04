import numpy as np
from models.network import NN

# XOR input data
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

# Create network with layer sizes
# input layer (size 2 for XOR) is implicitly defined by X.shape[0]
# hidden layer 1: 3 units, ReLU
# hidden layer 2: 3 units, ReLU
# output layer: 1 unit, Sigmoid
layer_sizes = [3, 3, 1]
activations = ['relu', 'relu', 'sigmoid']
network = NN(X=X, y=y, layer_sizes=layer_sizes, activations=activations)

# Train the network
print("Starting training...")
epochs = 5000
learning_rate = 0.05
epoch_losses = network.train(epochs=epochs, learning_rate=learning_rate)
print("Training finished.")

if epoch_losses:
    print(f"Final loss: {epoch_losses[-1]}")

# Test the network with XOR input data
print("\nTesting network predictions:")
predictions = network.predict(X)
print(f"Raw predictions: {predictions}")
rounded_predictions = np.round(predictions)
print(f"Rounded predictions: {rounded_predictions}")
print(f"Target (y): {y}")