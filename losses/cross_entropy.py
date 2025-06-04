import numpy as np

class BinaryCrossEntropyLoss:

    def loss(self, y_pred, y_true):
        m = y_true.shape[1]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (1/m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def loss_derivative(self, y_pred, y_true):
        m = y_true.shape[1]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (- y_true / y_pred + (1 - y_true) / (1 - y_pred)) / m
