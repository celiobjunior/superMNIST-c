#include "./headers/config.h"
#include "./headers/dataset.h"
#include "./headers/network.h"

#include <stdio.h>

int main(void)
{
        Network net;
        Dataset dataset = {0};
        size_t n_labels;
        f32 img[MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
        f32 learning_rate = LEARNING_RATE;

        network_init(&net, MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE);

        dataset_load_mnist_images(TRAIN_IMG_PATH,
                                  &dataset.images,
                                  &dataset.n_samples,
                                  &dataset.pixels_per_image);
        dataset_load_mnist_labels(TRAIN_LBL_PATH, &dataset.labels, &n_labels);

        if (dataset.n_samples != n_labels)
        {
                printf("Image/label count mismatch.\n");
                network_free(&net);
                dataset_free(&dataset);
                return 1;
        }

        if (dataset.pixels_per_image != (size_t)(MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE))
        {
                printf("Unexpected image dimensions.\n");
                network_free(&net);
                dataset_free(&dataset);
                return 1;
        }

        for (i32 epoch = 0; epoch < EPOCHS; epoch++)
        {
                printf("EPOCH #%d\n", epoch + 1);
                for (size_t i = 0; i < dataset.n_samples; i++)
                {
                        for (size_t j = 0; j < dataset.pixels_per_image; j++)
                                img[j] = dataset.images[i * dataset.pixels_per_image + j] / 255.0f;

                        network_train(&net, img, dataset.labels[i], learning_rate);
                }
        }

        network_free(&net);
        dataset_free(&dataset);

        return 0;
}
