# rawgrad

A from-scratch automatic differentiation engine written in pure C. No dependencies. No PyTorch. No Python. Just malloc, math, and gradients.

Implements a dynamic computation graph with reverse-mode autodiff, a full MLP layer stack, MSE and cross-entropy losses, SGD optimizer, and ships with two training demos: XOR and MNIST.

---

## What is this

Most people learn backprop by reading about it. This is what it looks like when you actually build it.

Every tensor tracks its own parents in a computation graph. Calling `tensor_backward()` performs a topological sort of that graph and invokes each node's backward kernel in reverse order. Gradients accumulate via the chain rule. That is the entire engine.

The rest of the codebase (linear layers, MLP, losses, optimizer) is built directly on top of those primitives with no external dependencies.

---

## Architecture

```
engine.c / engine.h       Core tensor type + all forward ops + autograd engine
nn.c / nn.h               Linear layer (weights + bias, forward pass, param access)
mlp.c / mlp.h             Multi-layer perceptron built from Linear layers
loss.c / loss.h           MSE loss, numerically stable fused softmax cross-entropy
optim.c / optim.h         SGD optimizer with zero_grad and lr scheduling
mnist_loader.c            Binary IDX-format MNIST reader
train_xor.c               XOR classification demo
train_mnist.c             Full MNIST training loop with batching and accuracy eval
```

---

## Tensor ops

Forward and backward kernels are implemented for:

| Op | Notes |
|---|---|
| `tensor_add` | Element-wise + bias broadcast (B, C) + (1, C) |
| `tensor_sub` | Element-wise subtraction |
| `tensor_mul` | Element-wise multiplication |
| `tensor_div` | Element-wise division |
| `tensor_pow` | Element-wise power, scalar exponent supported |
| `tensor_matmul` | (M, N) x (N, K) -> (M, K) with correct dA, dB |
| `tensor_expn` | Element-wise exp |
| `tensor_sqrt` | Element-wise sqrt |
| `tensor_mean` | Reduces to scalar, distributes gradient uniformly |
| `tensor_relu` | ReLU activation |
| `tensor_Tanh` | Tanh activation |
| `tensor_softmax` | Softmax (used inside fused cross-entropy) |

---

## Memory model

Tensors use manual reference counting. Every op that creates a new tensor retains its parents. Call `tensor_release()` on tensors you no longer hold a reference to. When the ref count hits zero, the entire subgraph is freed recursively.

```c
Tensor* z   = linear_forward(layer, x);
Tensor* a   = tensor_relu(z);
tensor_release(z);   // z is now owned by a's parent list
// ...
tensor_release(a);   // frees a and recursively z
```

No garbage collector. No arena. You know where your memory is.

---

## Build

Requires only `gcc` and `libm`.

```bash
make              # builds: train, train_xor, train_mnist
make clean
```

---

## Run

### XOR

Trains a 2-4-1 MLP with Tanh activations on the XOR problem using MSE loss. Converges in well under 5000 epochs.

```bash
./train_xor
```

```
Training XOR with MLP (2->4->1)...
Epoch 0, Loss: 0.254801
Epoch 500, Loss: 0.006412
...
Optimization Finished!
Predictions:
In: [0, 0] Target: 0 Pred: 0.0183
In: [0, 1] Target: 1 Pred: 0.9821
In: [1, 0] Target: 1 Pred: 0.9819
In: [1, 1] Target: 0 Pred: 0.0201
```

### MNIST

Trains a 784-256-128-10 MLP with ReLU activations on raw MNIST pixels. Cross-entropy loss with softmax, SGD, batched training.

Place the raw MNIST binary files in `data/`:
```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

Download from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) or any mirror.

```bash
./train_mnist
```

```
Loading MNIST data...
Loaded MNIST data: 60000 train, 10000 test
Model created. Parameters: 6
Training for 20 epochs | batch=32 | lr=0.0100
Epoch 1 [1850/1875] loss=0.3821 lr=0.01000
...
Final Test Accuracy: ~95%
```

---

## Cross-entropy implementation

The loss is implemented as a single fused op for numerical stability. It does not call `tensor_softmax` and then `tensor_log`. Instead, it computes log-softmax directly:

```c
log_prob = (logit - max) - log(sum_exp(logits - max))
loss     = -sum(target * log_prob) / batch_size
```

The backward pass recomputes softmax from the stored logits and sets:

```
dL/dz_j = (softmax_j - target_j) / batch_size
```

This is the standard analytically derived gradient of cross-entropy + softmax composed, implemented directly without going through the autograd graph for softmax itself.

---

## File map

```
engine.c          718 lines — the whole engine
nn.c              73 lines  — linear layer
mlp.c             91 lines  — MLP
loss.c            123 lines — MSE + cross-entropy
optim.c           43 lines  — SGD
mnist_loader.c    ~80 lines — IDX binary parser
train_xor.c       95 lines  — demo
train_mnist.c     203 lines — full training script
```

---

## License

MIT
