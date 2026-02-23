#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "engine.h"

// Helper to check approx equality
int close(float a, float b) {
    return fabs(a - b) < 1e-4;
}

void test_simple_add() {
    printf("Test: Simple Add... \n");
    printf("Creating a...\n");
    Tensor* a = tensor_create(2.0f);
    printf("Creating b...\n");
    Tensor* b = tensor_create(3.0f);
    printf("Creating c...\n");
    Tensor* c = tensor_add(a, b);
    
    // forward
    printf("Checking forward...\n");
    assert(close(*c->data, 5.0f));
    
    // backward
    printf("Backward...\n");
    tensor_backward(c);
    printf("Checking grads...\n");
    assert(close(*a->grad, 1.0f));
    assert(close(*b->grad, 1.0f));
    
    printf("Releasing...\n");
    tensor_release(c);
    tensor_release(a);
    tensor_release(b);
    
    printf("Passed\n");
}

void test_matrix_mul() {
    printf("Test: Matrix Mul... ");
    // A: 2x3, B: 3x2
    Tensor* A = tensor_create_matrix(2, 3);
    Tensor* B = tensor_create_matrix(3, 2);
    
    // Fill A with 1, B with 2
    for(int i=0; i<6; i++) A->data[i] = 1.0f;
    for(int i=0; i<6; i++) B->data[i] = 2.0f;
    
    Tensor* C = tensor_matmul(A, B);
    // C is 2x2. Each element is row(1,1,1) . col(2,2,2) = 1*2+1*2+1*2 = 6.
    assert(C->shape[0] == 2 && C->shape[1] == 2);
    for(int i=0; i<4; i++) assert(close(C->data[i], 6.0f));
    
    // Backward
    tensor_backward(C);
    
    for(int i=0; i<6; i++) {
        assert(close(A->grad[i], 4.0f));
    }

    tensor_release(C);
    tensor_release(A);
    tensor_release(B);
    printf("Passed\n");
}

void test_pow_bug() {
    printf("Test: Pow Bug... ");
    // x^2
    Tensor* x = tensor_create(3.0f);
    Tensor* two = tensor_create(2.0f);
    Tensor* y = tensor_pow(x, two);
    
    assert(close(*y->data, 9.0f));
    
    tensor_backward(y);
    // dy/dx = 2x = 6
    assert(close(*x->grad, 6.0f));
    
    tensor_release(y);
    tensor_release(x);
    tensor_release(two);
    printf("Passed\n");
}

void test_memory_stress() {
    printf("Test: Memory Stress (1000 iter)... ");
    for (int i=0; i<1000; i++) {
        Tensor* a = tensor_create(1.0f);
        Tensor* b = tensor_create(2.0f);
        Tensor* c = tensor_add(a, b);
        Tensor* d = tensor_mul(c, a);
        tensor_backward(d);
        
        tensor_release(d);
        tensor_release(c);
        tensor_release(b);
        tensor_release(a);
    }
    printf("Passed\n");
}

int main() {
    setbuf(stdout, NULL);
    test_simple_add();
    test_matrix_mul();
    test_pow_bug();
    test_memory_stress();
    
    printf("All tests passed!\n");
    return 0;
}
