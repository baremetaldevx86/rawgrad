#include "optim.h"
#include <stdlib.h>

SGD* sgd_create(Tensor** params, int n_params, float lr) {
    SGD* opt = (SGD*)malloc(sizeof(SGD));
    opt->params = params;
    opt->n_params = n_params;
    opt->lr = lr;
    return opt;
}

void sgd_step(SGD* opt) {
    for (int i = 0; i < opt->n_params; i++) {
        Tensor* p = opt->params[i];

        // scalar parameter
        if (p->size == 1) {
            *(p->data) -= opt->lr * (*(p->grad));
        }
        // matrix parameter
        else {
            for (int j = 0; j < p->size; j++) {
                p->data[j] -= opt->lr * p->grad[j];
            }
        }
    }
}

void sgd_zero_grad(SGD* opt) {
    for (int i = 0; i < opt->n_params; i++) {
        tensor_zero_grad(opt->params[i]);
    }
}

