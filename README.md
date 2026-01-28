# Autograd Engine in C
Built a minimal deep learning framework in C from scratch featuring a reverse-mode autograd engine, dynamic computation graphs, tensor operations with broadcasting, and end-to-end model training using SGD.ing linear regression models.

This project demonstrates how modern frameworks like PyTorch and TensorFlow work internally — from tensor algebra to backpropagation and optimization.

---

## Features

- Reverse-mode automatic differentiation (backpropagation)
- Dynamic computation graphs
- Tensor system with broadcasting support
- Matrix multiplication with gradient propagation
- Linear neural network layers
- Mean Squared Error (MSE) loss
- Stochastic Gradient Descent (SGD) optimizer
- End-to-end training loop
- Trains real models (e.g. linear regression)

---

## Architecture

The project is organized as a modular deep learning framework:

```text
autograd_engine/
├── engine.h # Tensor API and autograd interface
├── engine.c # Tensor engine + reverse-mode autodiff
├── nn.h # Neural network layers API
├── nn.c # Linear layer implementation
├── loss.h # Loss functions API
├── loss.c # MSE loss implementation
├── optim.h # Optimizer API
├── optim.c # SGD optimizer
├── main.c # Training loop
```

Each module mirrors how real ML frameworks are structured internally.

---

## What This Project Implements

This project builds the full training stack from scratch:

| Component | Description |
|---------|-------------|
Tensor Engine | Multi-dimensional tensor representation with gradients |
Autograd | Reverse-mode automatic differentiation |
Graph Engine | Dynamic computation graph construction |
Backprop | Topological traversal and gradient propagation |
Broadcasting | Bias broadcasting for neural layers |
Layers | Linear (fully-connected) layer |
Loss | Mean Squared Error (MSE) |
Optimizer | Stochastic Gradient Descent (SGD) |
Training | End-to-end model training loop |

---

## Example: Training Linear Regression

The framework is used to train a model to learn:

y = 3x + 2


Using a single-layer neural network:

y = Wx + b


### Sample Output

Epoch 0 | Loss = 5236.77
Epoch 50 | Loss = 0.92
Epoch 100 | Loss = 0.90
...
Epoch 450 | Loss = 0.75

Learned parameters:
W = 3.05
b = 0.31

Prediction for x=10: 30.82 (expected ≈ 32)


This confirms the engine correctly performs forward propagation, backpropagation, and optimization.

---

## 🛠 Build

Requires:
- GCC

Compile:

```bash
gcc engine.c nn.c loss.c optim.c main.c -o train -lm

▶ Run

./train
