#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE_IMAGE 28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define INPUT_SIZE 784
#define HIDDEN_SIZE 15
#define OUTPUT_SIZE 10

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct Layer {
        double *biases, *weights;
        int in_size, out_size;
}Layer;

typedef struct Network {
        Layer hidden, output;
        int num_layers;
} Network;

typedef struct Data {
        unsigned char *images, *labels;
        int n_imgs, img_size;
}Data;

void init(Layer *layer, int in_size, int out_size);

// NEURAL NETWORK FUNCTIONS

void feed_forward(Layer *layer, double *input, double *output);

// MATHEMATICAL FUNCTIONS

float sigmoid(int z);

// HELPERS

void shuffle(int *vec); // TODO

void read_mnist_imgs(const char *filename, unsigned char **images, int *n_imgs, int *img_size);

void read_mnist_labels(const char *filename, unsigned char **labels, int *n_labels);

int main(void) 
{
        Network net;
        Data data;

        init(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
        init(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

        read_mnist_imgs(TRAIN_IMG_PATH, &data.images, &data.n_imgs, &data.img_size);
        read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.n_imgs);

        free(net.hidden.weights);
        free(net.hidden.biases);
        free(net.output.weights);
        free(net.output.biases);
        free(data.images);
        free(data.labels);
}

void init(Layer *layer, int in_size, int out_size) 
{
        printf("Loading Neural Network...\n");
        int size = in_size * out_size;
        float scale = sqrtf(2.0f / in_size);

        layer->in_size = in_size;
        layer->out_size = out_size;

        layer->weights = (double *) malloc(size * sizeof(double));
        layer->biases = (double *) calloc(out_size, sizeof(double));

        for (int i = 0; i < size; i++)
                layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;

        printf("Neural Network loaded!\n\n");
}

void feed_forward(Layer *layer, double *input, double *output)
{
        // Initializing bias
        for (int i = 0; i < layer->out_size; i++) {
                output[i] = layer->biases[i];
        }

        // Matrix multiplication
        for(int i = 0; i < layer->in_size; i++) {
                for(int j = 0; j < layer->out_size; j++) {
                        output[j] += input[i] * layer->weights[i * layer->out_size + j];
                }
        }

        // Activation function - Sigmoid
        for (int i = 0; i < layer->out_size; i++) {
                output[i] = sigmoid(output[i]);
        }
}

float sigmoid(int z) 
{
        return 1.0 / (1.0 + exp(-z));
}

void read_mnist_imgs(const char *filename, unsigned char **images, int *n_imgs, int *img_size) 
{
        printf("Reading MNIST images...\n");

        int magic_number, n_rows, n_cols;
        FILE *F_IMGS = fopen(filename, "rb");

        if (!F_IMGS)
        {
                printf("The file does not exist!\n");
                exit(1);
        }

        fread(&magic_number, sizeof(int), 1, F_IMGS);
        
        fread(n_imgs, sizeof(int), 1, F_IMGS);
        *n_imgs = __builtin_bswap32(*n_imgs);
        fread(&n_rows, sizeof(int), 1, F_IMGS);
        n_rows = __builtin_bswap32(n_rows);
        fread(&n_cols, sizeof(int), 1, F_IMGS);
        n_cols = __builtin_bswap32(n_cols);
        
        *img_size = n_rows * n_cols;

        *images = (unsigned char *) malloc((*n_imgs) * (*img_size));
        fread(*images, sizeof(unsigned char), ((*n_imgs) * (*img_size)), F_IMGS);
        fclose(F_IMGS);

        printf("MNIST images completely loaded...\n\n");
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *n_labels)
{
        printf("Reading MNIST labels...\n");

        int magic_number;
        FILE *F_LABELS = fopen(filename, "rb");

        if (!F_LABELS)
        {
                printf("The file does not exist!\n");
                exit(1);
        }
        
        fread(&magic_number, sizeof(int), 1, F_LABELS);

        fread(n_labels, sizeof(int), 1, F_LABELS);
        *n_labels = __builtin_bswap32(*n_labels);

        *labels = (unsigned char *) malloc(*n_labels);
        fread(*labels, sizeof(unsigned char), (*n_labels), F_LABELS);
        fclose(F_LABELS);
        
        printf("MNIST labels completely loaded...\n\n");
}
