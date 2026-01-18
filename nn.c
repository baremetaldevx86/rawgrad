#include <bits/posix2_lim.h>
#include <stdlib.h>
#include "nn.h"
#include "engine.h"
#include "time.h"


// Utility: random float in [-1, 1]
static float rand_uniform() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

// Initialize weights with small random values
static void init_weights(Tensor* W) {
    for (int i = 0; i < W->size; i++) {
        W->data[i] = rand_uniform();
    }
}

// Initialize biases to zero
static void init_bias(Tensor* b) {
    for (int i = 0; i < b->size; i++) {
        b->data[i] = 0.0f;
    }
}

Linearlayer* linear_create(int in_features, int out_features) {
    srand(time(NULL));

    Linearlayer* layer = (Linearlayer*)malloc(sizeof(Linearlayer));
    
    layer->in_features = in_features;
    layer->out_features = out_features;

    // Weight matrix: (in_features x out_features)
    layer->W = tensor_create_matrix(in_features, out_features);

    // Bias vector: (1 x out_features)
    layer->b = tensor_create_matrix(1, out_features);

    // Initialize parameters
    init_weights(layer->W);
    init_bias(layer->b); 
    
    return layer;
}

// Forward pass: y = x @ W + b
Tensor* linear_forward(Linearlayer* layer, Tensor* x) {
    Tensor* y = tensor_matmul(x, layer->W);
    y = tensor_add(y, layer->b);
    return y;
} 

// Return parameters for optimizer
Tensor** linear_params(Linearlayer* layer, int* n_params) {
    Tensor** params = (Tensor**)malloc(sizeof(Tensor*) * 2);
    params[0] = layer->W;
    params[1] = layer->b;
    *n_params = 2;
    return params;
}
