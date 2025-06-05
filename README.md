# 🧠 Neural Network from Scratch

A flexible, modular implementation of a fully-connected neural network built **from scratch using NumPy**.

---

## 🚀 Features

- ✅ **Multi-layer Neural Network**: Define architectures with any number of layers and units.
- 🧩 **Multiple Activation Functions**: Supports `Sigmoid`, `ReLU`, and `Tanh`.
- ⚙️ **Weight Initialization**: Uses Xavier for sigmoid/tanh and He for ReLU.
- ⛓ **Mini-batch Gradient Descent**: Configurable batch sizes for efficient training.
- 📊 **Binary Cross-Entropy Loss**: Ideal for binary classification tasks.
- 🧠 **Mathematically Correct Backpropagation**: Gradient computation is cleanly separated from parameter updates.

---

## 📂 Project Structure

```plaintext
neural-network-scratch/
├── models/
│   ├── net_layer.py          # Layer class: forward/backward propagation
│   └── neural_network.py     # NeuralNetwork class (configurable & correct)
├── optimizers/
│   └── mini_batch_sgd.py     # Mini-batch SGD optimizer
├── losses/
│   └── cross_entropy.py      # Binary cross-entropy loss
├── example_usage.py          # Example training & evaluation script
├── main.py                   # Entry point to run your custom setup
├── requirements.txt          # Dependencies
└── README.md                 # You're here
```

---

## 🔧 Components

### 1. `Layer` – `models/net_layer.py`
The core building block of the network:
- Proper weight initialization (Xavier/He)
- Forward pass: computes `Z = WX + b`, applies activation
- Backward pass: uses chain rule for gradients
- Supports `Sigmoid`, `ReLU`, and `Tanh`

### 2. `NeuralNetwork` – `neural_network.py`
Manages architecture, training, and inference:
- Define custom architectures via layer sizes and activations
- Training loop with correct gradient flow
- Prediction and evaluation support

### 3. `Optimizer` – `optimizers/mini_batch_sgd.py`
Implements Mini-batch SGD:
- Random shuffling and batching
- Parameter updates via gradient descent
- Configurable learning rate

### 4. `Loss` – `losses/cross_entropy.py`
Binary cross-entropy loss function:
- Computes classification loss
- Provides gradient for output layer during backprop

---

## 🧪 Usage

Run the example usage script:

```bash
python main.py
````


---

## 📦 Installation

```bash
pip install -r requirements.txt
```

> Only NumPy is required.

---

## 📚 Example

```python
from neural_network import NeuralNetwork

nn = NeuralNetwork(
    layer_sizes=[2, 4, 1],
    activations=['relu', 'sigmoid'],
    learning_rate=0.1,
    batch_size=2,
    epochs=1000
)

nn.train(X_train, y_train)
preds = nn.predict(X_test)
```


## 🧠 Why This Project?

To truly understand deep learning, building a neural network from the ground up is invaluable. This project gives you control over every detail—activations, gradients, losses, and updates—so you can learn by doing.

---


