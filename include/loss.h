// loss = mean ((y_pred - y_true)**2)
// loss = -sum(y_true * log(softmax(y_pred)))
#ifndef LOSS_H
#define LOSS_H

#include "engine.h"

Tensor* mse_loss(Tensor* y_pred, Tensor* y_true);

// Logits: (batch_size, n_classes) - raw scores
// Targets: (batch_size, n_classes) - one-hot encoded, or probabilities
Tensor* cross_entropy_loss(Tensor* logits, Tensor* targets);

#endif
