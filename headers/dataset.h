#ifndef DATASET_H
#define DATASET_H

#include "types.h"
#include <stddef.h>

typedef struct Dataset {
        u8 *images;
        u8 *labels;
        size_t n_samples;
        size_t pixels_per_image;
} Dataset;

void dataset_load_mnist_images(const char *filename,
                               u8 **images,
                               size_t *n_samples,
                               size_t *pixels_per_image);

void dataset_load_mnist_labels(const char *filename,
                               u8 **labels,
                               size_t *n_labels);

void dataset_free(Dataset *dataset);

#endif
