#ifndef HELPERS_H
#define HELPERS_H

#include "types.h"
#include <stddef.h>

typedef struct Data {
        u8 *images, *labels;
        size_t n_samples, pixels_per_image;
}Data;


void shuffle(i32 *vec);  // TODO

/**
 * @brief Reads label data from an MNIST IDX1 file.
 *
 * The MNIST label file contains an 8-byte header followed by one byte per
 * label. All 32-bit header fields are stored in big-endian order (MSB first):
 * - bytes 0..3 : magic number
 * - bytes 4..7 : number of labels
 * - bytes 8..  : label data
 *
 * Label organization in the file:
 * - each label is stored as one `u8`
 * - valid label range is `0..9`
 * - labels are stored sequentially, with no padding
 * - label for image `i` is stored at `(*labels)[i]`
 *
 * @param filename Path to the MNIST IDX1 label file.
 * @param labels Output parameter that receives a heap-allocated contiguous
 *               buffer containing all labels. The caller must free `*labels`
 *               with `free()`.
 * @param n_labels Output parameter that receives the total number of labels
 *                 stored in the file header.
 */
void read_mnist_labels(const char *filename, u8 **labels, size_t *n_labels);

/**
 * @brief Reads image data from an MNIST IDX3 file.
 *
 * The MNIST image file contains a 16-byte header followed by raw pixel bytes.
 * All 32-bit header fields are stored in big-endian order (MSB first):
 * - bytes 0..3   : magic number
 * - bytes 4..7   : number of images
 * - bytes 8..11  : number of rows per image
 * - bytes 12..15 : number of columns per image
 * - bytes 16..   : pixel data
 *
 * Pixel organization in the file:
 * - each pixel is stored as one `u8`
 * - valid pixel range is `0..255`
 * - pixels are stored row-wise inside each image
 * - images are stored sequentially, with no padding between them
 *
 * Buffer organization after loading:
 * - `*pixels_per_image = rows * cols`
 * - image `i` starts at offset `i * (*pixels_per_image)`
 * - pixel `(row, col)` from image `i` is stored at:
 *   `(*images)[i * (*pixels_per_image) + row * cols + col]`
 *
 * @param filename Path to the MNIST IDX3 image file.
 * @param images Output parameter that receives a heap-allocated contiguous
 *               buffer containing all pixels from all images. The caller must
 *               free `*images` with `free()`.
 * @param n_samples Output parameter that receives the total number of images
 *                  stored in the file header.
 * @param pixels_per_image Output parameter that receives the number of pixels
 *                         per image (`rows * cols`).
 */
void read_mnist_imgs(const char *filename, u8 **images, size_t *n_samples, size_t *pixels_per_image);

#endif // HELPERS_H
