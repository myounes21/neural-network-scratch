import numpy as np
from models.network import NN
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
data = load_breast_cancer()
X_full = data.data
y_full = data.target

# Select only two features for simplicity (radius and texture)
X = X_full[:, [0, 1]].T  # shape: (2, n_samples)
y = y_full.reshape(1, -1)  # shape: (1, n_samples)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X.T).T

# Create and train network
layer_sizes = [3, 2, 1]  # 2 inputs -> 3 hidden -> 2 hidden -> 1 output
activations = ['relu', 'relu', 'sigmoid']
network = NN(X=X, y=y, layer_sizes=layer_sizes, activations=activations)

# Train
print("Training on breast cancer data...")
epochs = 1000
learning_rate = 0.01
losses = network.train(epochs=epochs, learning_rate=learning_rate)

# Test predictions
predictions = network.predict(X)
accuracy = np.mean((predictions > 0.5) == y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")