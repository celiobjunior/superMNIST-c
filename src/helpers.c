#include "../headers/helpers.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/**
 * @brief Reads a 32-bit unsigned integer stored in big-endian order.
 *
 * This helper reads four bytes from an MNIST IDX header and assembles them
 * into a native `u32` value, independent of the host machine endianness.
 *
 * @param file Binary stream positioned at the start of the 4-byte field.
 * @return The decoded 32-bit unsigned value.
 */
static u32 read_be_u32(FILE *file) {
    u8 bytes[4];

    if (fread(bytes, sizeof(u8), 4, file) != 4) {
        printf("Failed to read MNIST header.\n");
        exit(1);
    }

    return ((u32) bytes[0] << 24) |
           ((u32) bytes[1] << 16) |
           ((u32) bytes[2] << 8)  |
           ((u32) bytes[3]);
}

void read_mnist_imgs(const char *filename, u8 **images, size_t *n_samples, size_t *pixels_per_image)
{
        printf("Reading MNIST images...\n");

        u32 magic_number, n_images_u32, n_rows, n_cols, pixels_per_image_u32;
        size_t total_bytes;
        FILE *F_IMGS = fopen(filename, "rb");

        if (!F_IMGS)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        magic_number = read_be_u32(F_IMGS);
        n_images_u32 = read_be_u32(F_IMGS);
        n_rows = read_be_u32(F_IMGS);
        n_cols = read_be_u32(F_IMGS);

        if (magic_number != 2051U)
        {
                printf("Invalid MNIST image file.\n");
                fclose(F_IMGS);
                exit(1);
        }

        if (n_images_u32 > (u32) INT32_MAX || n_rows > (u32) INT32_MAX || n_cols > (u32) INT32_MAX)
        {
                printf("MNIST image file contains unsupported dimensions.\n");
                fclose(F_IMGS);
                exit(1);
        }

        if (n_rows != 0 && n_cols > ((u32) INT32_MAX / n_rows))
        {
                printf("MNIST image size is too large.\n");
                fclose(F_IMGS);
                exit(1);
        }

        pixels_per_image_u32 = n_rows * n_cols; // 784 for 28x28 images

        if (pixels_per_image_u32 != 0 && n_images_u32 > ((u32) SIZE_MAX / pixels_per_image_u32))
        {
                printf("MNIST image buffer is too large.\n");
                fclose(F_IMGS);
                exit(1);
        }

        *n_samples = (size_t) n_images_u32;
        *pixels_per_image = (size_t) pixels_per_image_u32;
        total_bytes = (size_t) n_images_u32 * (size_t) pixels_per_image_u32;

        *images = malloc(total_bytes);
        fread(*images, sizeof(u8), total_bytes, F_IMGS);
        fclose(F_IMGS);

        printf("MNIST images completely loaded...\n\n");
}

void read_mnist_labels(const char *filename, u8 **labels, size_t *n_labels)
{
        printf("Reading MNIST labels...\n");

        u32 magic_number, n_labels_u32;
        FILE *F_LABELS = fopen(filename, "rb");

        if (!F_LABELS)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        magic_number = read_be_u32(F_LABELS);
        n_labels_u32 = read_be_u32(F_LABELS);

        if (magic_number != 2049U)
        {
                printf("Invalid MNIST label file.\n");
                fclose(F_LABELS);
                exit(1);
        }

        if (n_labels_u32 > (u32) INT32_MAX)
        {
                printf("MNIST label file contains too many labels.\n");
                fclose(F_LABELS);
                exit(1);
        }

        *n_labels = (size_t) n_labels_u32;

        *labels = malloc((size_t) n_labels_u32);
        fread(*labels, sizeof(u8), (size_t) n_labels_u32, F_LABELS);
        fclose(F_LABELS);

        printf("MNIST labels completely loaded...\n\n");
}
