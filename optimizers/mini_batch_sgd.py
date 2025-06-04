import numpy as np

class MiniBatchSGD:
    def __init__(self, lr=0.01, max_itr=1000, tol=1e-6, batch_size=32):
        self.lr = lr
        self.max_itr = max_itr
        self.tol = tol
        self.batch_size = batch_size
        self.losses = []

    def get_batches(self, X, y):
        """Generate mini-batches from the data"""
        m = X.shape[1]

        if m == 0:
            return  # Or yield nothing: yield from ()

        if m < self.batch_size:
            yield X, y
        else:
            indices = np.random.permutation(m)
            n_batches = m // self.batch_size

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                yield X[:, batch_indices], y[:, batch_indices]

            if m % self.batch_size != 0:
                start_idx = n_batches * self.batch_size
                batch_indices = indices[start_idx:]
                yield X[:, batch_indices], y[:, batch_indices]

    def step(self,layer, dW, db):
        """Update layer parameters"""
        layer._W -= self.lr * dW
        layer._b -= self.lr * db
        return layer._W, layer._b