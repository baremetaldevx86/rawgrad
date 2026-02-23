// loss = mean ((y_pred - y_true)**2)
#include "engine.h"

Tensor* mse_loss(Tensor* y_pred, Tensor* y_true) {
    Tensor* diff = tensor_sub(y_pred, y_true); 
    Tensor* sq = tensor_mul(diff, diff);
    Tensor* loss = tensor_mean(sq);
    
    // Release local references to intermediates
    tensor_release(diff);
    tensor_release(sq);
    
    return loss;
}

// ============================================================
// Cross Entropy Loss (with Softmax)
// ============================================================
// Implemented as a single fused operation for numerical stability
// Loss = -sum(target * log(softmax(logits))) / batch_size

static void cross_entropy_backward(Tensor* loss) {
    Tensor* logits = loss->parents[0];
    Tensor* targets = loss->parents[1];
    
    int batch_size = logits->shape[0];
    int n_classes = logits->shape[1];
    
    // We need to recompute Softmax(logits) to calculate gradients
    // grad_logit = (softmax - target) / batch_size
    
    // Note: In a production engine, we might cache the softmax output in the forward pass
    // to avoid recomputing here. For simplicity/memory, we recompute.
    
    for (int i = 0; i < batch_size; i++) {
        // 1. Find max for stability
        float max_val = -1e9f;
        for (int j = 0; j < n_classes; j++) {
            float v = logits->data[i*n_classes + j];
            if (v > max_val) max_val = v;
        }
        
        // 2. Compute exp and sum
        float sum_exp = 0.0f;
        // We can use a temporary buffer or just compute twice. 
        // Let's compute sum first.
        for (int j = 0; j < n_classes; j++) {
            sum_exp += exp(logits->data[i*n_classes + j] - max_val);
        }
        
        // 3. Compute gradient
        // dL/dz_j = (p_j - y_j) / N
        for (int j = 0; j < n_classes; j++) {
            float p_j = exp(logits->data[i*n_classes + j] - max_val) / sum_exp;
            float y_j = targets->data[i*n_classes + j];
            
            // Apply gradient from loss (usually 1.0)
            float update = (p_j - y_j) * loss->grad[0] / (float)batch_size; // mean reduction
            
            logits->grad[i*n_classes + j] += update;
        }
    }
}

Tensor* cross_entropy_loss(Tensor* logits, Tensor* targets) {
    // Inputs: (B, C)
    // Output: Scalar (mean loss)
    
    if (logits->ndim != 2 || targets->ndim != 2) {
        printf("Cross Entropy expects 2D inputs (batch, classes)\n");
        exit(1);
    }
    
    int batch_size = logits->shape[0];
    int n_classes = logits->shape[1];
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < batch_size; i++) {
        // Softmax normalization per row (Stable: subtract max)
        
        // 1. Find max x per row
        float max_val = -1e9f;
        for (int j = 0; j < n_classes; j++) {
            float v = logits->data[i*n_classes + j];
            if (v > max_val) max_val = v;
        }
        
        // 2. Sum exp(x - max)
        float sum_exp = 0.0f;
        for (int j = 0; j < n_classes; j++) {
            // logits->data[...] access
            sum_exp += exp(logits->data[i*n_classes + j] - max_val);
        }
        
        // 3. Log Softmax = (x - max) - log(sum_exp)
        float log_sum_exp = log(sum_exp);
        
        // 4. Cross Entropy: -SUM(target * log_prob)
        for (int j = 0; j < n_classes; j++) {
            float log_prob = (logits->data[i*n_classes + j] - max_val) - log_sum_exp;
            float target = targets->data[i*n_classes + j];
            total_loss -= target * log_prob;
        }
    }
    
    // Mean reduction
    Tensor* loss = tensor_create(total_loss / (float)batch_size);
    
    // Wire graph
    loss->n_parents = 2;
    loss->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    loss->parents[0] = logits;
    loss->parents[1] = targets;
    tensor_retain(logits);
    tensor_retain(targets);
    
    loss->backward = cross_entropy_backward;
    
    return loss;
}

