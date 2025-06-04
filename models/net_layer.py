import numpy as np

class Layer:
    def __init__(self, n_units, activation="sigmoid"):
        self.n_units = n_units
        self.activation_name = activation
        self._W = None
        self._b = None
        self.cache = {}

    def initialize(self, input_dim):
        self._W = np.random.randn(self.n_units, input_dim)
        self._b = np.zeros((self.n_units, 1))

    def forward(self, A_prev):
        Z = self._W @ A_prev + self._b
        A = self._activate(Z)
        self.cache = {"A_prev": A_prev, "Z": Z, "A": A}
        return A

    def backward(self, dA):
        A_prev = self.cache["A_prev"]
        Z = self.cache["Z"]
        m = A_prev.shape[1]

        dL_dA = dA  # gradient of loss w.r.t activation output A
        dA_dZ = self._activation_derivative(Z)  # derivative of activation function A w.r.t Z

        dL_dZ = dL_dA * dA_dZ  # chain rule: dL/dZ = dL/dA * dA/dZ

        dL_dW = (1 / m) * dL_dZ @ A_prev.T  # gradient of loss w.r.t weights W
        dL_db = (1 / m) * np.sum(dL_dZ, axis=1, keepdims=True)  # gradient of loss w.r.t biases b

        dL_dA_prev = self._W.T @ dL_dZ  # gradient of loss w.r.t previous activations A_prev

        return dL_dA_prev, dL_dW, dL_db

    def _activate(self, Z):
        if self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        # add more activations here
        elif self.activation_name == "relu":
            return np.maximum(0, Z)
        elif self.activation_name == "tanh":
            return np.tanh(Z)
        else:
            raise NotImplementedError(f"Activation '{self.activation_name}' not implemented")

    def _activation_derivative(self, Z):
        if self.activation_name == "sigmoid":
            sig = 1 / (1 + np.exp(-Z))
            return sig * (1 - sig)
        elif self.activation_name == "relu":
            return (Z > 0).astype(float)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(Z) ** 2
        else:
            raise NotImplementedError(f"Derivative for '{self.activation_name}' not implemented")

