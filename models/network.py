import numpy as np
from joblib.testing import param

from models.net_layer import Layer  # use correct pathfrom optimizers.mini_batch_sgd import MiniBatchSGD
from losses.cross_entropy import BinaryCrossEntropyLoss

#
# class NN:
#
#     def __init__(self, layers, X, y):
#         self.layers = layers
#         self.X = X
#         self.y = y
#
#
#
#         layers = [2, 3, 3, 1]
#
#         layer = Layer()
#         self._W = layer._W
#         self._b = layer._b
#
#         loss = BinaryCrossEntropyLoss()
#         optimizer = MiniBatchSGD()
#
#         for l in layers:
#            layer.n_units = l
#            self.X, self.y = optimizer.get_batches(X, y)
#
#            y_pred =  layer.forward(self.X)
#            dA = loss.loss_derivative(y_true=self.y, y_pred=y_pred)
#            dA_prev, dW, db = layer.backward(dA)
#
#            self._W , self._b = optimizer.step(layer, dW, db)

class NN:
    def __init__(self, layer_sizes, X, y):
        self.X = X
        self.y = y
        self.layers = []

        # Create layers with proper initialization
        input_dim = X.shape[0]
        for n_units in layer_sizes:
            layer = Layer(n_units=n_units)
            layer.initialize(input_dim)
            self.layers.append(layer)
            input_dim = n_units

        self.loss = BinaryCrossEntropyLoss()
        self.optimizer = MiniBatchSGD()
