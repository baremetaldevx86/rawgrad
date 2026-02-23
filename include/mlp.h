#ifndef MLP_H
#define MLP_H

#include "engine.h"
#include "nn.h"

typedef struct {
    Linearlayer** layers;
    int n_layers;
} MLP;

// Create an MLP with specified layer sizes
// layer_sizes: array of integers defining input/output sizes [in, h1, h2, ..., out]
// n_layers: number of layers (length of layer_sizes - 1)
MLP* mlp_create(int* layer_sizes, int n_layers);

// Forward pass
// use_relu: 1 for ReLU activation, 0 for Tanh
Tensor* mlp_forward(MLP* mlp, Tensor* x, int use_relu);

// Get all parameters from all layers
Tensor** mlp_params(MLP* mlp, int* n_params);

// Free MLP memory
void mlp_free(MLP* mlp);

#endif
