#ifndef NETWORK_H
#define NETWORK_H

#include "types.h"
#include <stddef.h>

#define HIDDEN_LAYER_SIZE 15
#define OUTPUT_LAYER_SIZE 10

typedef struct Layer {
        f32 *biases, *weights;
        size_t input_count, output_count;
}Layer;

typedef struct Network {
        Layer hidden, output;
        i32 num_layers;
} Network;

void init(Layer *layer, size_t input_count, size_t output_count);

void train(Network *net, const f32 *input, u8 label, f32 learning_rate);

// NEURAL NETWORK FUNCTIONS

void feed_forward(const Layer *layer, const f32 *input, f32 *output);

void backprop(Network *net, const f32 *input, const f32 *hidden_output, const f32 *final_output, u8 label, f32 learning_rate);

// MATHEMATICAL FUNCTIONS

f32 sigmoid(f32 z);

f32 d_sigmoid(f32 z);

#endif
