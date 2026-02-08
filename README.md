# C Autograd Engine

A lightweight, tensor-based autograd engine implemented in C. Supports dynamic computation graphs, automatic differentiation, and basic neural network layers.

## Features

- **Tensors**: Scalar and Matrix support (2D).
- **Autograd**: Reverse-mode automatic differentiation (Backpropagation).
- **Operations**: Add, Sub, Mul, Pow, Exp, Tanh, ReLU, MatMul, CrossEntropy.
- **Neural Nets**: Linear layers, MLP (Multi-Layer Perceptron).
- **Memory Management**: Reference counting for efficient memory usage.
- **Optimizer**: Stochastic Gradient Descent (SGD).

## Usage

### Building

```bash
make
make train_xor
make train_mnist
```

### Example: Training MNIST

1. Download MNIST dataset (https://github.com/cvdfoundation/mnist?tab=readme-ov-file)
2. Extract files to `data/` directory:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`
3. Run training:
```bash
./train_mnist
```

### Example: Training a Linear Model

```c
#include "engine.h"
#include "nn.h"
#include "loss.h"
#include "optim.h"

int main() {
    // 1. Create tensors
    Tensor* x = tensor_create_matrix(10, 1);
    // ... populate data ...

    // 2. Define model
    Linearlayer* model = linear_create(1, 1);

    // 3. Forward pass
    Tensor* y = linear_forward(model, x);

    // 4. Compute loss
    Tensor* loss = mse_loss(y, target);

    // 5. Backward pass
    tensor_backward(loss);

    // 6. Update weights
    // ...

    // 7. Cleanup (Crucial!)
    tensor_release(loss); // Releases the entire graph for this iteration
    tensor_release(y);    // Release local reference
    // ...
}
```

## Memory Management

Each tensor has a reference count.
- `tensor_create*`: Returns tensor with ref_count = 1.
- `tensor_retain(t)`: Increments ref_count.
- `tensor_release(t)`: Decrements ref_count. If 0, frees memory and releases parents.

**Rule**: If you create a tensor or get one from an operation (which creates a new one), you own a reference. You must release it when done. If you pass it to another operation, that operation retains it, so you still hold your reference.

## Testing

```bash
make
./train
```

To run the test suite:
```bash
gcc -Wall -g -O2 -o test_engine test_engine.c engine.c -lm
./test_engine
```
