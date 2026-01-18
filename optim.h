#ifndef OPTIM_H
#define OPTIM_H

#include "engine.h"

typedef struct {
    Tensor** params;
    int n_params;
    float lr;
} SGD;

SGD* sgd_create(Tensor** params, int n_params, float lr);
void sgd_step(SGD* opt);
void sgd_zero_grad(SGD* opt);

#endif
