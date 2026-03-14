#include "../headers/dataset.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static u32 read_be_u32(FILE *file)
{
        u8 bytes[4];

        if (fread(bytes, sizeof(u8), 4, file) != 4)
        {
                printf("Failed to read MNIST header.\n");
                exit(1);
        }

        return ((u32) bytes[0] << 24) |
               ((u32) bytes[1] << 16) |
               ((u32) bytes[2] << 8)  |
               ((u32) bytes[3]);
}

void dataset_free(Dataset *dataset)
{
        if (!dataset)
                return;

        free(dataset->images);
        free(dataset->labels);

        dataset->images = NULL;
        dataset->labels = NULL;
        dataset->n_samples = 0;
        dataset->pixels_per_image = 0;
}

void dataset_load_mnist_images(const char *filename,
                               u8 **images,
                               size_t *n_samples,
                               size_t *pixels_per_image)
{
        printf("Reading MNIST images...\n");

        u32 magic_number;
        u32 n_images_u32;
        u32 n_rows;
        u32 n_cols;
        u32 pixels_per_image_u32;
        size_t total_bytes;
        FILE *images_file = fopen(filename, "rb");

        if (!images_file)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        magic_number = read_be_u32(images_file);
        n_images_u32 = read_be_u32(images_file);
        n_rows = read_be_u32(images_file);
        n_cols = read_be_u32(images_file);

        if (magic_number != 2051U)
        {
                printf("Invalid MNIST image file.\n");
                fclose(images_file);
                exit(1);
        }

        if (n_images_u32 > (u32) INT32_MAX ||
            n_rows > (u32) INT32_MAX ||
            n_cols > (u32) INT32_MAX)
        {
                printf("MNIST image file contains unsupported dimensions.\n");
                fclose(images_file);
                exit(1);
        }

        if (n_rows != 0 && n_cols > ((u32) INT32_MAX / n_rows))
        {
                printf("MNIST image size is too large.\n");
                fclose(images_file);
                exit(1);
        }

        pixels_per_image_u32 = n_rows * n_cols;

        if (pixels_per_image_u32 != 0 &&
            n_images_u32 > ((u32) SIZE_MAX / pixels_per_image_u32))
        {
                printf("MNIST image buffer is too large.\n");
                fclose(images_file);
                exit(1);
        }

        *n_samples = (size_t) n_images_u32;
        *pixels_per_image = (size_t) pixels_per_image_u32;
        total_bytes = (size_t) n_images_u32 * (size_t) pixels_per_image_u32;

        *images = malloc(total_bytes);
        if (!*images)
        {
                printf("Failed to allocate MNIST image buffer.\n");
                fclose(images_file);
                exit(1);
        }

        if (fread(*images, sizeof(u8), total_bytes, images_file) != total_bytes)
        {
                printf("Failed to read MNIST image data.\n");
                fclose(images_file);
                free(*images);
                *images = NULL;
                exit(1);
        }

        fclose(images_file);

        printf("MNIST images completely loaded...\n\n");
}

void dataset_load_mnist_labels(const char *filename, u8 **labels, size_t *n_labels)
{
        printf("Reading MNIST labels...\n");

        u32 magic_number;
        u32 n_labels_u32;
        FILE *labels_file = fopen(filename, "rb");

        if (!labels_file)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        magic_number = read_be_u32(labels_file);
        n_labels_u32 = read_be_u32(labels_file);

        if (magic_number != 2049U)
        {
                printf("Invalid MNIST label file.\n");
                fclose(labels_file);
                exit(1);
        }

        if (n_labels_u32 > (u32) INT32_MAX)
        {
                printf("MNIST label file contains too many labels.\n");
                fclose(labels_file);
                exit(1);
        }

        *n_labels = (size_t) n_labels_u32;
        *labels = malloc((size_t) n_labels_u32);

        if (!*labels)
        {
                printf("Failed to allocate MNIST label buffer.\n");
                fclose(labels_file);
                exit(1);
        }

        if (fread(*labels, sizeof(u8), (size_t) n_labels_u32, labels_file) != (size_t) n_labels_u32)
        {
                printf("Failed to read MNIST label data.\n");
                fclose(labels_file);
                free(*labels);
                *labels = NULL;
                exit(1);
        }

        fclose(labels_file);

        printf("MNIST labels completely loaded...\n\n");
}
