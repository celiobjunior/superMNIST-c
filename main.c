#include "./headers/main.h"
#include "./headers/helpers.h"
#include "./headers/network.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
        Network net;
        Data data = {0};
        size_t n_labels;
        f32 img[MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        f32 learning_rate = LEARNING_RATE;

        init(&net.hidden, MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE, HIDDEN_LAYER_SIZE);
        init(&net.output, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

        read_mnist_imgs(TRAIN_IMG_PATH, &data.images, &data.n_samples, &data.pixels_per_image);
        read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &n_labels);

        if (data.n_samples != n_labels)
        {
                printf("Image/label count mismatch.\n");
                free(net.hidden.weights);
                free(net.hidden.biases);
                free(net.output.weights);
                free(net.output.biases);
                free(data.images);
                free(data.labels);
                return 1;
        }

        if (data.pixels_per_image != (size_t) (MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE))
        {
                printf("Unexpected image dimensions.\n");
                free(net.hidden.weights);
                free(net.hidden.biases);
                free(net.output.weights);
                free(net.output.biases);
                free(data.images);
                free(data.labels);
                return 1;
        }

        for(i32 epoch = 0; epoch < EPOCHS; epoch++)
        {
                printf("EPOCH #%d\n", epoch + 1);
                for(size_t i = 0; i < data.n_samples; i++)
                {
                        for(size_t j = 0; j < data.pixels_per_image; j++)
                        {
                                img[j] = data.images[i * data.pixels_per_image + j] / 255.0f;
                        }
                        train(&net, img, data.labels[i], learning_rate);
                }
        }

        free(net.hidden.weights);
        free(net.hidden.biases);
        free(net.output.weights);
        free(net.output.biases);
        free(data.images);
        free(data.labels);
}
