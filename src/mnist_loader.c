#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist_loader.h"

// Helper: Read 32-bit integer (Big Endian -> Host Endian)
static uint32_t read_uint32(FILE* f) {
    uint32_t v;
    if (fread(&v, sizeof(v), 1, f) != 1) return 0;
    // Swap bytes: data is Big Endian, x86 is Little Endian
    return ((v << 24) & 0xFF000000) |
           ((v << 8)  & 0x00FF0000) |
           ((v >> 8)  & 0x0000FF00) |
           ((v >> 24) & 0x000000FF);
}

MNISTData* load_mnist(const char* images_path, const char* labels_path) {
    FILE* f_img = fopen(images_path, "rb");
    FILE* f_lbl = fopen(labels_path, "rb");

    if (!f_img || !f_lbl) {
        printf("Error: Could not open MNIST files: %s, %s\n", images_path, labels_path);
        if (f_img) fclose(f_img);
        if (f_lbl) fclose(f_lbl);
        return NULL;
    }

    // Read headers
    uint32_t magic_img = read_uint32(f_img);
    uint32_t n_img = read_uint32(f_img);
    uint32_t rows = read_uint32(f_img);
    uint32_t cols = read_uint32(f_img);

    uint32_t magic_lbl = read_uint32(f_lbl);
    uint32_t n_lbl = read_uint32(f_lbl);

    if (magic_img != 2051 || magic_lbl != 2049 || n_img != n_lbl) {
        printf("Error: Invalid MNIST file format or count mismatch\n");
        printf("Magic: %d / %d, Count: %d / %d\n", magic_img, magic_lbl, n_img, n_lbl);
        fclose(f_img);
        fclose(f_lbl);
        return NULL;
    }

    MNISTData* data = (MNISTData*)malloc(sizeof(MNISTData));
    data->n_samples = n_img;
    data->input_dim = rows * cols; // 28 * 28 = 784

    // Allocate memory
    data->images = (float*)malloc(sizeof(float) * data->n_samples * data->input_dim);
    data->labels = (int*)malloc(sizeof(int) * data->n_samples);

    // Read Data
    // Images: Read byte by byte and normalize to [0, 1]
    unsigned char* img_buf = (unsigned char*)malloc(data->input_dim);
    for (int i = 0; i < data->n_samples; i++) {
        fread(img_buf, 1, data->input_dim, f_img);
        for (int j = 0; j < data->input_dim; j++) {
            data->images[i * data->input_dim + j] = (float)img_buf[j] / 255.0f;
        }
    }
    free(img_buf);

    // Labels: Read byte by byte
    unsigned char lbl_buf;
    for (int i = 0; i < data->n_samples; i++) {
        fread(&lbl_buf, 1, 1, f_lbl);
        data->labels[i] = (int)lbl_buf;
    }

    fclose(f_img);
    fclose(f_lbl);

    printf("Loaded MNIST data: %d samples, %d features\n", data->n_samples, data->input_dim);
    return data;
}

void mnist_free(MNISTData* data) {
    if (data) {
        free(data->images);
        free(data->labels);
        free(data);
    }
}
