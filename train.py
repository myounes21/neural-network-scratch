import numpy as np
from models.network import NN

# XOR input data
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

# Create network with layer sizes
layer_sizes = [2, 3, 3, 1]  # input layer is handled automatically
network = NN(layer_sizes, X, y)