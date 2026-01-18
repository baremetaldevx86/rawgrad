// loss = mean ((y_pred - y_true)**2)
#include "engine.h"

Tensor* mse_loss(Tensor* y_pred, Tensor* y_true) {
    Tensor* diff = tensor_sub(y_pred, y_true); 
    Tensor* loss = tensor_mean(tensor_mul(diff, diff));
    return loss;
}

