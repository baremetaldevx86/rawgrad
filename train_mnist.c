#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "engine.h"
#include "nn.h"
#include "mlp.h"
#include "loss.h"
#include "optim.h"
#include "mnist_loader.h"

// Configuration
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01f
#define EPOCHS 20

// Helper: Get accuracy
float compute_accuracy(MLP* model, MNISTData* data, int n_samples) {
    int correct = 0;
    
    // Evaluate in batches to save memory
    int eval_batch_size = 100;
    
    Tensor* X_batch = tensor_create_matrix(eval_batch_size, 784);
    
    for (int i = 0; i < n_samples; i += eval_batch_size) {
        int current_batch_size = (i + eval_batch_size > n_samples) ? (n_samples - i) : eval_batch_size;
        
        // Load batch
        // Note: For simplicity, we just reuse the fixed size tensor and ignore the tail garbage if batch is smaller
        // In a real engine, we'd resize. Here we just keycare about the valid rows.
        if (current_batch_size != eval_batch_size) {
            // Lazy: Skip tail for accuracy check or handle properly
            // Let's just break for simplicity in this C example
            break; 
        }

        for (int b = 0; b < current_batch_size; b++) {
            for (int k = 0; k < 784; k++) {
                X_batch->data[b*784 + k] = data->images[(i + b) * 784 + k];
            }
        }
        
        Tensor* logits = mlp_forward(model, X_batch, 1); // use_relu = 1
        
        // Argmax
        for (int b = 0; b < current_batch_size; b++) {
            float max_val = -1e9f;
            int max_idx = -1;
            for (int k = 0; k < 10; k++) {
                if (logits->data[b*10 + k] > max_val) {
                    max_val = logits->data[b*10 + k];
                    max_idx = k;
                }
            }
            
            if (max_idx == data->labels[i + b]) {
                correct++;
            }
        }
        
        tensor_release(logits);
    }
    
    tensor_release(X_batch);
    
    return (float)correct / (float)n_samples; // Approximate if we skipped tail
}

int main() {
    srand(time(NULL));
    
    // 1. Load Data
    printf("Loading MNIST data...\n");
    MNISTData* train_data = load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MNISTData* test_data = load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    
    if (!train_data || !test_data) {
        printf("Failed to load data. Make sure 'data' folder exists and contains unpacked MNIST files.\n");
        return 1;
    }
    
    // 2. Create Model: 784 -> 128 -> 10
    int layer_sizes[] = {784, 256, 128, 10};
    MLP* model = mlp_create(layer_sizes, 3);
    
    // 3. Optimizer
    int n_params;
    Tensor** params = mlp_params(model, &n_params);
    SGD* opt = sgd_create(params, n_params, LEARNING_RATE);
    
    printf("Model created. Parameters: %d\n", n_params);
    printf("Starting training for %d epochs (Batch size: %d, LR: %f)...\n", EPOCHS, BATCH_SIZE, LEARNING_RATE);
    
    // 4. Training Loop
    Tensor* X_batch = tensor_create_matrix(BATCH_SIZE, 784);
    Tensor* Y_batch = tensor_create_matrix(BATCH_SIZE, 10); // One-hot
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        int n_batches = train_data->n_samples / BATCH_SIZE;
        
        for (int b = 0; b < n_batches; b++) {
            // Prepare batch
            int start_idx = b * BATCH_SIZE;
            
            // Fill X
            for (int i = 0; i < BATCH_SIZE; i++) {
                for (int k = 0; k < 784; k++) {
                    X_batch->data[i*784 + k] = train_data->images[(start_idx + i) * 784 + k];
                }
            }
            
            // Fill Y (One-hot)
            for (int i = 0; i < BATCH_SIZE * 10; i++) Y_batch->data[i] = 0.0f;
            for (int i = 0; i < BATCH_SIZE; i++) {
                int label = train_data->labels[start_idx + i];
                Y_batch->data[i*10 + label] = 1.0f;
            }
            
            // Forward
            Tensor* logits = mlp_forward(model, X_batch, 1);
            
            // Loss
            Tensor* loss = cross_entropy_loss(logits, Y_batch);
            epoch_loss += loss->data[0];
            
            // Backward
            sgd_zero_grad(opt);
            tensor_backward(loss);
            sgd_step(opt);
            
            // Cleanup
            tensor_release(logits);
            tensor_release(loss);
            
            if (b % 100 == 0) {
                printf("\rEpoch %d [%d/%d] Loss: %.4f", epoch + 1, b, n_batches, epoch_loss / (b+1));
                fflush(stdout);
            }
        }
        
        printf("\nEpoch %d finished. Avg Loss: %.4f\n", epoch + 1, epoch_loss / n_batches);
        
        // Evaluate
        // float acc = compute_accuracy(model, test_data, 1000); // Check first 1000 for speed
        // printf("Test Accuracy (subset): %.2f%%\n", acc * 100.0f);
    }
    
    // Final Test
    printf("Computing final accuracy...\n");
    float final_acc = compute_accuracy(model, test_data, test_data->n_samples);
    printf("Final Test Accuracy: %.2f%%\n", final_acc * 100.0f);
    
    // Cleanup
    tensor_release(X_batch);
    tensor_release(Y_batch);
    mnist_free(train_data);
    mnist_free(test_data);
    mlp_free(model);
    sgd_free(opt);
    free(params);
    
    return 0;
}
