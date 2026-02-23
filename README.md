# C Autograd Engine

A lightweight, from-scratch autograd engine written in C.  
This project implements dynamic computation graphs, reverse-mode automatic differentiation, and a minimal neural network stack without relying on external ML libraries.

The goal is to understand how modern deep learning systems work at the lowest level: tensor ownership, graph construction, backpropagation, and optimization.

---

## Features

### Core
- Tensor system supporting scalars and 2D matrices
- Dynamic computation graphs built at runtime
- Reverse-mode automatic differentiation (backpropagation)
- Explicit and deterministic memory management

### Operations
- Elementwise: Add, Sub, Mul, Pow, Exp
- Activations: ReLU, Tanh
- Linear algebra: MatMul
- Loss functions: CrossEntropy, MSE

### Neural Networks
- Linear (fully connected) layers
- Multi-Layer Perceptron (MLP)
- ReLU-based hidden layers
- One-hot classification support

### Optimization
- Stochastic Gradient Descent (SGD)
- Manual learning-rate control
- Clean separation of model and optimizer ownership

### Memory Management
- Reference counting (no garbage collector)
- Explicit graph teardown after each iteration
- Valgrind- and ASAN-clean execution

---

## Project Structure

```c
.
├── engine.c / engine.h # Tensor and autograd core
├── nn.c / nn.h # Neural network primitives
├── mlp.c / mlp.h # MLP implementation
├── loss.c / loss.h # Loss functions
├── optim.c / optim.h # Optimizers (SGD)
├── mnist_loader.c / .h # MNIST data loader
├── train.c # Basic training demo
├── train_xor.c # XOR classification example
├── train_mnist.c # MNIST training (MLP)
├── test_engine.c # Unit tests
├── Makefile # Build system
└── data/ # MNIST dataset (not tracked)
```

---

## Building

Requirements:
- GCC or Clang
- make
- Standard C math library (`-lm`)

Build all targets:

make


Build specific binaries:

make train
make train_xor
make train_mnist


Clean build artifacts:

make clean


---

## Training on MNIST

### Dataset Setup

Download the MNIST dataset from:

https://github.com/cvdfoundation/mnist


Extract the following files into a `data/` directory:


data/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte


### Run Training


./train_mnist


Example output:

Model created. Parameters: 6
Training for 20 epochs | batch=32 | lr=0.0050
Epoch 20 finished. Avg loss: 0.18
Final Test Accuracy: 96.2%


---

## Other Examples

### XOR Classification


./train_xor


This example demonstrates:
- Nonlinear decision boundaries
- Correctness of end-to-end autograd and backpropagation

---

## Minimal Training Loop Example

```c
Tensor* x = tensor_create_matrix(32, 784);
Tensor* y = tensor_create_matrix(32, 10);

Tensor* logits = mlp_forward(model, x, 1);
Tensor* loss   = cross_entropy_loss(logits, y);

sgd_zero_grad(opt);
tensor_backward(loss);
sgd_step(opt);

/* Cleanup: releases the entire computation graph */
tensor_release(loss);
tensor_release(logits);

Rule: If you create a tensor or receive one from an operation, you own a reference and must release it.

Memory Management Model

Every tensor has a reference count

Ownership is explicit and deterministic

Rules

tensor_create*() returns a tensor with ref_count = 1

tensor_retain(t) increments the reference count

tensor_release(t) decrements the reference count

When ref_count == 0, the tensor is freed and its parents are released recursively

Design Philosophy

No garbage collector

No hidden frees

Graph lifetime is fully controlled by the user

Testing

Run basic training test:

./train

Run engine unit tests:

gcc -Wall -g -O2 -o test_engine test_engine.c engine.c -lm
./test_engine

The project is free of memory leaks, double frees, and use-after-free errors.

Project Goals

This project is not intended to compete with full-featured frameworks such as PyTorch or TensorFlow.

It is designed to:

Understand the mechanics of automatic differentiation

Practice low-level ML systems design

Explore explicit memory ownership in graph-based computation

Serve as a base for further experimentation

Future Work

Momentum and Adam optimizers

Inference-only forward path

Convolutional neural networks

Computation graph visualization

Model save and load support

Dataset augmentation and diagnostics

License

MIT License
