#include "engine.h"
#include <stdlib.h>

// ============================================================
// Internal helpers
// ============================================================

static void add_backward(Tensor* self);
static void sub_backward(Tensor* self);
static void mean_backward(Tensor* self);
static void mul_backward(Tensor* self);
static void pow_backward(Tensor* self);
static void exp_backward(Tensor* self);
static void tanh_backward(Tensor* self);
static void matmul_backward(Tensor* self);

// -------------------- Dynamic list for topo -----------------

typedef struct {
    Tensor** items;
    int size;
    int capacity;
} TensorList;

static TensorList* list_create() {
    TensorList* l = (TensorList*)malloc(sizeof(TensorList));
    l->capacity = 32;
    l->size = 0;
    l->items = (Tensor**)malloc(sizeof(Tensor*) * l->capacity);
    return l;
}

static void list_push(TensorList* l, Tensor* t) {
    if (l->size == l->capacity) {
        l->capacity *= 2;
        l->items = (Tensor**)realloc(l->items, sizeof(Tensor*) * l->capacity);
    }
    l->items[l->size++] = t;
}

static int is_visited(Tensor** visited, int count, Tensor* t) {
    for (int i = 0; i < count; i++) {
        if (visited[i] == t) return 1;
    }
    return 0;
}

static void build_topo(Tensor* v, TensorList* topo, Tensor** visited, int* visited_count) {
    if (is_visited(visited, *visited_count, v)) return;

    visited[(*visited_count)++] = v;

    for (int i = 0; i < v->n_parents; i++) {
        build_topo(v->parents[i], topo, visited, visited_count);
    }

    list_push(topo, v);
}

// ============================================================
// Tensor creation
// ============================================================

Tensor* tensor_create(float x) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));

    t->ndim = 0;
    t->shape = NULL;
    t->size = 1;

    t->data = (float*)malloc(sizeof(float));
    t->grad = (float*)malloc(sizeof(float));

    *(t->data) = x;
    *(t->grad) = 0.0f;

    t->parents = NULL;
    t->n_parents = 0;
    t->backward = NULL;

    return t;
}

Tensor* tensor_create_matrix(int rows, int cols) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));

    t->ndim = 2;
    t->shape = (int*)malloc(2 * sizeof(int));
    t->shape[0] = rows;
    t->shape[1] = cols;
    t->size = rows * cols;

    t->data = (float*)malloc(sizeof(float) * t->size);
    t->grad = (float*)malloc(sizeof(float) * t->size);

    for (int i = 0; i < t->size; i++) {
        t->grad[i] = 0.0f;
    }

    t->parents = NULL;
    t->n_parents = 0;
    t->backward = NULL;

    return t;
}


// ============================================================
// Backward kernels
// ============================================================


static void add_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];

    
    // Case 1: same shape
    if (a->size == b->size) {
        for (int i = 0; i < self->size; i++) {
            a->grad[i] += self->grad[i];
            b->grad[i] += self->grad[i];
        }
    }
    // Case 2: bias broadcast
    else if (b->size == a->shape[1]) {
        int rows = a->shape[0];
        int cols = a->shape[1];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                a->grad[i*cols + j] += self->grad[i*cols + j];
                b->grad[j] += self->grad[i*cols + j];
            }
        }
    } 
}

static void sub_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];

    for (int i = 0; i < self->size; i++) {
        a->grad[i] += self->grad[i];
        b->grad[i] -= self->grad[i];
    }
}

static void mul_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];

    for (int i = 0; i < self->size; i++) {
        a->grad[i] += b->data[i] * self->grad[i];
        b->grad[i] += a->data[i] * self->grad[i];
    }
}

static void mean_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    float g = self->grad[0] / (float)a->size;

    for (int i = 0; i < a->size; i++) {
        a->grad[i] += g;
    }
}

static void pow_backward(Tensor* self) {
    Tensor* a = self->parents[0];
    Tensor* b = self->parents[1];

    float av = *(a->data);
    float bv = *(b->data);
    float cv = *(self->data);
    float grad = *(self->grad);

    *(a->grad) += bv * pow(av, bv - 1.0f) * grad;
    *(b->grad) += cv * log(av) * grad;
}

static void exp_backward(Tensor* self) {
    Tensor* a = self->parents[0];

    float cv = *(self->data);
    float grad = *(self->grad);

    *(a->grad) += cv * grad;
}

static void tanh_backward(Tensor* self) {
    Tensor* a = self->parents[0];

    float cv = *(self->data);
    float grad = *(self->grad);

    *(a->grad) += (1.0f - cv * cv) * grad;
}

static void matmul_backward(Tensor* C) {
    Tensor* A = C->parents[0];
    Tensor* B = C->parents[1];

    int m = A->shape[0];
    int n = A->shape[1];
    int k = B->shape[1];

    // dA = dC @ B^T
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += C->grad[i*k + p] * B->data[j*k + p];
            }
            A->grad[i*n + j] += sum;
        }
    }

    // dB = A^T @ dC
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < m; p++) {
                sum += A->data[p*n + i] * C->grad[p*k + j];
            }
            B->grad[i*k + j] += sum;
        }
    }
}



// ============================================================
// Forward ops
// ============================================================

Tensor* tensor_add(Tensor* a, Tensor* b) {
    Tensor* c = tensor_create_matrix(a->shape[0], a->shape[1]);
    
    // same shape 
    if(a->size == b->size) {
        for (int i = 0; i < a->size; i++) {
            c->data[i] = a->data[i] + b->data[i];
        }
    }
    // broadcast bias (1 x out)
    else if(b->size == a->shape[1]) {
        int rows = a->shape[0];
        int cols = a->shape[1];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                c->data[i*cols + j] = a->data[i*cols + j] + b->data[j];
            }
        }
    }
    else {
        printf("tensor_add shape mismatch\n");
        exit(1);
    }


    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;
    c->n_parents = 2;
    c->backward = add_backward;
    return c;
} 

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    Tensor* c = tensor_create_matrix(a->shape[0], a->shape[1]);

    for (int i = 0; i < a->size; i++) {
        c->data[i] = a->data[i] - b->data[i];
    }

    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;
    c->n_parents = 2;
    c->backward = sub_backward;
    return c;
}

Tensor* tensor_mean(Tensor *a) {
    // compute sum
    float sum = 0.0f;
    for (int i = 0; i < a->size; i++) {
        sum += a->data[i];
    }

    // output is a scalar tensor
    Tensor* c = tensor_create(sum / (float)a->size);

    // wire graph
    c->n_parents = 1;
    c->parents = (Tensor**)malloc(sizeof(Tensor*));
    c->parents[0] = a;
    c->backward = mean_backward;

    return c;
}


Tensor* tensor_mul(Tensor* a, Tensor* b) {

    Tensor* c = tensor_create_matrix(a->shape[0], a->shape[1]);
  
    for (int i = 0; i < a->size; i++) {
        c->data[i] = a->data[i] * b->data[i];
    }

    c->n_parents = 2;
    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;

    c->backward = mul_backward;
    return c;
}

Tensor* tensor_pow(Tensor* a, Tensor* b) {
    Tensor* c = tensor_create(pow(*(a->data), *(b->data)));

    c->n_parents = 2;
    c->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    c->parents[0] = a;
    c->parents[1] = b;

    c->backward = pow_backward;
    return c;
}

Tensor* tensor_expn(Tensor* a) {
    Tensor* c = tensor_create(exp(*(a->data)));

    c->n_parents = 1;
    c->parents = (Tensor**)malloc(sizeof(Tensor*));
    c->parents[0] = a;

    c->backward = exp_backward;
    return c;
}

Tensor* tensor_Tanh(Tensor* a) {
    Tensor* c = tensor_create(tanh(*(a->data)));

    c->n_parents = 1;
    c->parents = (Tensor**)malloc(sizeof(Tensor*));
    c->parents[0] = a;

    c->backward = tanh_backward;
    return c;
}

// -------------------- Matrix multiplication -----------------

Tensor* tensor_matmul(Tensor* A, Tensor* B) {
    int m = A->shape[0];
    int n = A->shape[1];
    int k = B->shape[1];

    if (B->shape[0] != n) {
        printf("Matmul shape mismatch\n");
        exit(1);
    }

    Tensor* C = tensor_create_matrix(m, k);

    // forward
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int p = 0; p < n; p++) {
                sum += A->data[i*n + p] * B->data[p*k + j];
            }
            C->data[i*k + j] = sum;
        }
    }

    C->n_parents = 2;
    C->parents = (Tensor**)malloc(sizeof(Tensor*) * 2);
    C->parents[0] = A;
    C->parents[1] = B;
    C->backward = matmul_backward;

    return C;
}
// ============================================================
// Autograd engine
// ============================================================

void tensor_backward(Tensor* t) {
    TensorList* topo = list_create();

    Tensor* visited[4096];
    int visited_count = 0;

    build_topo(t, topo, visited, &visited_count);

    // seed gradient
    if (t->size == 1) {
        *(t->grad) = 1.0f;
    } else {
        for (int i = 0; i < t->size; i++) {
            t->grad[i] = 1.0f;
        }
    }

    for (int i = topo->size - 1; i >= 0; i--) {
        Tensor* node = topo->items[i];
        if (node->backward) {
            node->backward(node);
        }
    }
}

// ============================================================
// Utilities
// ============================================================

void tensor_zero_grad(Tensor* t) {
    if (t->size == 1) {
        *(t->grad) = 0.0f;
    } else {
        for (int i = 0; i < t->size; i++) {
            t->grad[i] = 0.0f;
        }
    }
}
 
