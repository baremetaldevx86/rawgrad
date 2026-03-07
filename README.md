# rawgrad

A from-scratch automatic differentiation engine written in pure C. No frameworks. No dependencies. No Python. Just raw C, manual memory management, and reverse-mode autodiff.

Trains a 3-layer MLP on MNIST from raw pixels and reaches **98.01% test accuracy** -- all in ~1,150 lines of library code.

<img width="823" height="780" alt="mnist-training" src="https://github.com/user-attachments/assets/bc408345-be08-4b17-a631-2288f440e64d" />

---

## Why This Exists

Every deep learning framework hides the autograd behind thousands of lines of abstraction. rawgrad does not.

The entire differentiation engine fits in a single 715-line C file. Every tensor tracks its parents in a dynamically constructed computation graph. Calling `tensor_backward()` topologically sorts that graph and walks it in reverse, invoking each node's backward kernel. Gradients accumulate via the chain rule. That is the whole engine.

Everything else -- linear layers, MLP, loss functions, the optimizer -- is built directly on top of those primitives.

---

## Features

- **Reverse-mode automatic differentiation** with dynamic computation graph
- **13 differentiable operations** with hand-written forward and backward kernels
- **Matrix operations** including batched matmul with correct gradient computation
- **Fused softmax cross-entropy loss** for numerical stability (log-sum-exp trick)
- **He weight initialization** (Kaiming) via Box-Muller normal sampling
- **Manual reference-counted memory management** -- no GC, no arenas
- **SGD optimizer** with step-decay learning rate scheduling
- **MNIST data loader** that reads raw IDX binary format
- **Zero external dependencies** -- only `gcc` and `libm`

---

## Results

### MNIST

Architecture: `784 -> 256 -> 128 -> 10` (ReLU activations, softmax cross-entropy loss)

| Metric | Value |
|---|---|
| Test Accuracy | **98.01%** |
| Parameters | 235,146 |
| Optimizer | SGD (lr=0.02, step decay) |
| Batch Size | 32 |
| Epochs | 20 |

### XOR

Architecture: `2 -> 4 -> 1` (Tanh activation, MSE loss)

Converges to near-zero loss in 5,000 epochs, correctly classifying all four XOR inputs.

---

## Architecture

```
src/
  engine.c           715 lines  -- core tensor type, forward/backward ops, autograd
  nn.c                88 lines  -- linear layer (y = xW + b), He init
  mlp.c               99 lines  -- multi-layer perceptron
  loss.c              122 lines  -- MSE loss, fused softmax cross-entropy
  optim.c              42 lines  -- SGD optimizer
  mnist_loader.c       84 lines  -- IDX binary format parser

include/
  engine.h  nn.h  mlp.h  loss.h  optim.h  mnist_loader.h

examples/
  train.c             89 lines  -- minimal training demo
  train_xor.c         94 lines  -- XOR classification
  train_mnist.c      204 lines  -- full MNIST training loop

tests/
  test.c              12 lines
  test_engine.c      112 lines
```

---

## Supported Operations

Every operation has both a forward kernel and a hand-written backward kernel.

| Operation | Signature | Backward |
|---|---|---|
| Addition | `tensor_add(a, b)` | dL/da += dL/dc, dL/db += dL/dc (with bias broadcast) |
| Subtraction | `tensor_sub(a, b)` | dL/da += dL/dc, dL/db -= dL/dc |
| Multiplication | `tensor_mul(a, b)` | dL/da += b * dL/dc, dL/db += a * dL/dc |
| Division | `tensor_div(a, b)` | dL/da += dL/dc / b, dL/db += -a * dL/dc / b^2 |
| Power | `tensor_pow(a, b)` | Chain rule with scalar exponent support |
| Matrix Multiply | `tensor_matmul(A, B)` | dA = dC @ B^T, dB = A^T @ dC |
| Exponential | `tensor_expn(a)` | dL/da += exp(a) * dL/dc |
| Square Root | `tensor_sqrt(a)` | dL/da += dL/dc / (2 * sqrt(a)) |
| Mean | `tensor_mean(a)` | dL/da_i += dL/dc / n |
| ReLU | `tensor_relu(a)` | dL/da += dL/dc if a > 0, else 0 |
| Tanh | `tensor_Tanh(a)` | dL/da += (1 - tanh^2(a)) * dL/dc |
| Softmax | `tensor_softmax(a)` | Used inside fused cross-entropy |
| Cross-Entropy | `cross_entropy_loss(logits, targets)` | dL/dz = (softmax(z) - y) / batch_size |

---

## How the Autograd Works

```
1. Forward pass builds a DAG of Tensor nodes
2. Each operation creates a new Tensor that stores pointers to its parents
3. tensor_backward() is called on the loss node:
   a. Topological sort of the full computation graph
   b. Seed the loss gradient to 1.0
   c. Walk nodes in reverse topological order
   d. Each node calls its backward function pointer
   e. Gradients accumulate into parent .grad buffers via the chain rule
```

The topological sort uses a recursive DFS with a visited set. The backward pass is a single linear sweep over the sorted list.

---

## Memory Model

Tensors use **manual reference counting**. Every operation that creates a new tensor calls `tensor_retain()` on its parents. When you are done with a tensor, call `tensor_release()`. When the reference count hits zero, the tensor and its entire unreachable subgraph are freed recursively.

```c
Tensor* z = linear_forward(layer, x);    // z retains x internally
Tensor* a = tensor_relu(z);              // a retains z internally
tensor_release(z);                        // safe: z is still held by a
// ... use a ...
tensor_release(a);                        // frees a, then z (ref count -> 0)
```

No garbage collector. No arena allocator. You own every allocation.

---

## Cross-Entropy Implementation

The cross-entropy loss is implemented as a **single fused operation** rather than composing `softmax` and `log` through the autograd graph. This avoids numerical instability from computing `log(softmax(x))` naively.

**Forward:**
```
log_prob = (logit - max_logit) - log(sum(exp(logits - max_logit)))
loss     = -sum(target * log_prob) / batch_size
```

**Backward:**

The backward pass recomputes softmax from the stored logits and directly applies the analytically derived gradient:

```
dL/dz_j = (softmax(z_j) - target_j) / batch_size
```

This is the standard composed derivative of cross-entropy with softmax, bypassing the autograd graph for the softmax step itself.

---

## Build

Requires only `gcc` and `libm`. No `cmake`, no `configure`, no package manager.

```bash
make                # builds: train, train_xor, train_mnist
make clean          # removes all build artifacts
```

---

## Run

### MNIST

Place the raw MNIST IDX binary files in `data/`:

```
data/train-images-idx3-ubyte
data/train-labels-idx1-ubyte
data/t10k-images-idx3-ubyte
data/t10k-labels-idx1-ubyte
```

Download from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) or any mirror. Then:

```bash
./train_mnist
```

```
Loading MNIST data...
Loaded MNIST data: 60000 train, 10000 test
Model created. Parameters: 235146
Training for 20 epochs | batch=32 | lr=0.0200
Epoch 1 [1850/1875] loss=0.7218 lr=0.02000
...
Epoch 20 finished. Avg loss: 0.0089
Computing final accuracy...
Final Test Accuracy: 98.01%
```

### XOR

```bash
./train_xor
```

```
Training XOR with MLP (2->4->1)...
Epoch 0, Loss: 0.286245
Epoch 500, Loss: 0.005137
...
Optimization Finished!
Predictions:
In: [0, 0] Target: 0 Pred: 0.0042
In: [0, 1] Target: 1 Pred: 0.9871
In: [1, 0] Target: 1 Pred: 0.9863
In: [1, 1] Target: 0 Pred: 0.0184
```

---

## API Overview

### Tensor

```c
Tensor* t = tensor_create(3.14f);                  // scalar
Tensor* m = tensor_create_matrix(32, 784);          // 2D matrix
tensor_retain(t);                                    // increment ref count
tensor_release(t);                                   // decrement, free if zero
tensor_backward(loss);                               // run reverse-mode autodiff
tensor_zero_grad(t);                                 // reset gradients to zero
```

### Neural Network

```c
Linearlayer* fc = linear_create(784, 256);           // y = xW + b
Tensor* out = linear_forward(fc, input);             // forward pass

int sizes[] = {784, 256, 128, 10};
MLP* model = mlp_create(sizes, 3);                   // 3-layer MLP
Tensor* logits = mlp_forward(model, x, 1);           // 1 = ReLU, 0 = Tanh
```

### Loss

```c
Tensor* loss = mse_loss(predicted, target);          // mean squared error
Tensor* loss = cross_entropy_loss(logits, one_hot);  // fused softmax + CE
```

### Optimizer

```c
SGD* opt = sgd_create(params, n_params, 0.02f);
sgd_zero_grad(opt);                                  // zero all param gradients
tensor_backward(loss);                                // compute gradients
sgd_step(opt);                                        // w -= lr * grad
sgd_set_lr(opt, 0.002f);                             // learning rate decay
```

---

## License

MIT
