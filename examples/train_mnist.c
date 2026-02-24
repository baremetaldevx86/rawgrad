#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "engine.h"
#include "nn.h"
#include "mlp.h"
#include "loss.h"
#include "optim.h"
#include "mnist_loader.h"

/* ================= CONFIG ================= */

#define BATCH_SIZE     32
#define BASE_LR        0.02f
#define EPOCHS         20
#define EVAL_BATCH     100
#define SEED           42

/* ========================================= */

/* Shuffle index array (Fisher–Yates) */
static void shuffle_indices(int* idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

/* Compute accuracy on arbitrary dataset */
static float compute_accuracy(MLP* model, MNISTData* data) {
    int correct = 0;
    int n = data->n_samples;

    Tensor* X = tensor_create_matrix(EVAL_BATCH, 784);

    for (int i = 0; i < n; i += EVAL_BATCH) {
        int bs = (i + EVAL_BATCH > n) ? (n - i) : EVAL_BATCH;

        /* Fill batch */
        for (int b = 0; b < bs; b++) {
            for (int k = 0; k < 784; k++) {
                X->data[b * 784 + k] =
                    data->images[(i + b) * 784 + k];
            }
        }

        /* Forward (no backward called → safe) */
        Tensor* logits = mlp_forward(model, X, 1);

        for (int b = 0; b < bs; b++) {
            int pred = 0;
            float maxv = logits->data[b * 10];

            for (int k = 1; k < 10; k++) {
                float v = logits->data[b * 10 + k];
                if (v > maxv) {
                    maxv = v;
                    pred = k;
                }
            }

            if (pred == data->labels[i + b])
                correct++;
        }

        tensor_release(logits);
    }

    tensor_release(X);
    return (float)correct / (float)n;
}

/* =================== MAIN =================== */

int main(void) {
    srand(SEED);

    printf("Loading MNIST data...\n");

    MNISTData* train =
        load_mnist("data/train-images-idx3-ubyte",
                   "data/train-labels-idx1-ubyte");

    MNISTData* test =
        load_mnist("data/t10k-images-idx3-ubyte",
                   "data/t10k-labels-idx1-ubyte");

    if (!train || !test) {
        fprintf(stderr, "Failed to load MNIST data\n");
        return 1;
    }

    printf("Loaded MNIST data: %d train, %d test\n",
           train->n_samples, test->n_samples);

    /* Model: 784 → 256 → 128 → 10 */
    int layers[] = {784, 256, 128, 10};
    MLP* model = mlp_create(layers, 3);

    int n_params = 0;
    Tensor** params = mlp_params(model, &n_params);

    SGD* opt = sgd_create(params, n_params, BASE_LR);

    printf("Model created. Parameters: %d\n", mlp_count_scalar_params(model));
    printf("Training for %d epochs | batch=%d | lr=%.4f\n",
           EPOCHS, BATCH_SIZE, BASE_LR);

    /* Training buffers */
    Tensor* X = tensor_create_matrix(BATCH_SIZE, 784);
    Tensor* Y = tensor_create_matrix(BATCH_SIZE, 10);

    /* Shuffle indices */
    int* indices = malloc(train->n_samples * sizeof(int));
    for (int i = 0; i < train->n_samples; i++)
        indices[i] = i;

    /* ================= TRAIN LOOP ================= */

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        /* LR decay */
        float lr = BASE_LR;
        //if (epoch >= 10) lr *= 0.01f;
        //else if (epoch >= 5) lr *= 0.1f;
        sgd_set_lr(opt, lr);

        shuffle_indices(indices, train->n_samples);

        float epoch_loss = 0.0f;
        int n_batches = train->n_samples / BATCH_SIZE;

        for (int b = 0; b < n_batches; b++) {
            int base = b * BATCH_SIZE;

            /* X */
            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = indices[base + i];
                for (int k = 0; k < 784; k++) {
                    X->data[i * 784 + k] =
                        train->images[idx * 784 + k];
                }
            }

            /* Y (one-hot) */
            for (int i = 0; i < BATCH_SIZE * 10; i++)
                Y->data[i] = 0.0f;

            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = indices[base + i];
                Y->data[i * 10 + train->labels[idx]] = 1.0f;
            }

            Tensor* logits = mlp_forward(model, X, 1);
            Tensor* loss = cross_entropy_loss(logits, Y);

            epoch_loss += loss->data[0] * BATCH_SIZE;

            sgd_zero_grad(opt);
            tensor_backward(loss);
            sgd_step(opt);

            tensor_release(logits);
            tensor_release(loss);

            if (b % 100 == 0) {
                printf("\rEpoch %d [%d/%d] loss=%.4f lr=%.5f",
                       epoch + 1, b, n_batches,
                       epoch_loss / ((b + 1) * BATCH_SIZE),
                       lr);
                fflush(stdout);
            }
        }

        epoch_loss /= (n_batches * BATCH_SIZE);
        printf("\nEpoch %d finished. Avg loss: %.4f\n",
               epoch + 1, epoch_loss);
    }

    /* ================= EVAL ================= */

    printf("Computing final accuracy...\n");
    float acc = compute_accuracy(model, test);
    printf("Final Test Accuracy: %.2f%%\n", acc * 100.0f);

    /* ================= CLEANUP ================= */

    free(indices);
    tensor_release(X);
    tensor_release(Y);
    sgd_free(opt);
    free(params);
    mlp_free(model);
    mnist_free(train);
    mnist_free(test);

    return 0;
}
