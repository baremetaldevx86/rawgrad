#include <stdlib.h>
#include <stdio.h>
#include "mlp.h"

MLP* mlp_create(int* layer_sizes, int n_layers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->n_layers = n_layers;
    mlp->layers = (Linearlayer**)malloc(sizeof(Linearlayer*) * n_layers);

    for (int i = 0; i < n_layers; i++) {
        // layer_sizes[i] is input size, layer_sizes[i+1] is output size
        mlp->layers[i] = linear_create(layer_sizes[i], layer_sizes[i+1]);
    }

    return mlp;
}

Tensor* mlp_forward(MLP* mlp, Tensor* x, int use_relu) {
    Tensor* current_input = x;
    tensor_retain(current_input); // Retain initial input since we release inside loop

    for (int i = 0; i < mlp->n_layers; i++) {
        // Linear transformation
        Tensor* z = linear_forward(mlp->layers[i], current_input);
        
        // Release previous input (we are done with it for valid graph construction handling)
        // Note: linear_forward retains ‘current_input’ as a parent of ‘z’, 
        // so we can release our local holding reference.
        tensor_release(current_input);
        
        // Apply activation if not the last layer
        // (Last layer usually feeds into loss directly, e.g. MSE or CrossEntropyLogits)
        if (i < mlp->n_layers - 1) {
            Tensor* a;
            if (use_relu) {
                a = tensor_relu(z);
            } else {
                a = tensor_Tanh(z);
            }
            tensor_release(z); // z is now parent of a
            current_input = a;
        } else {
            // Last layer: linear output (logits)
            current_input = z;
        }
    }

    return current_input;
}

Tensor** mlp_params(MLP* mlp, int* n_params) {
    // First pass: count total parameters
    int total_params = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        int layer_n_params;
        Tensor** layer_params = linear_params(mlp->layers[i], &layer_n_params);
        total_params += layer_n_params;
        free(layer_params); // free the temporary array, not the tensors
    }

    // Allocate array
    Tensor** params = (Tensor**)malloc(sizeof(Tensor*) * total_params);
    
    // Second pass: collect parameters
    int offset = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        int layer_n_params;
        Tensor** layer_params = linear_params(mlp->layers[i], &layer_n_params);
        
        for (int j = 0; j < layer_n_params; j++) {
            params[offset + j] = layer_params[j];
        }
        offset += layer_n_params;
        
        free(layer_params);
    }

    *n_params = total_params;
    return params;
}

void mlp_free(MLP* mlp) {
    if (mlp) {
        for (int i = 0; i < mlp->n_layers; i++) {
            linear_free(mlp->layers[i]);
        }
        free(mlp->layers);
        free(mlp);
    }
}
