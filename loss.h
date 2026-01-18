#ifndef LOSS_H
#define LOSS_H

#include "engine.h"

Tensor* mse_loss(Tensor* y_pred, Tensor* y_true);

#endif
