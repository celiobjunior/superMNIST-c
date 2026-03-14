#ifndef NETWORK_H
#define NETWORK_H

#include "types.h"
#include <stddef.h>

typedef struct Layer {
        f32 *biases, *weights;
        size_t input_count, output_count;
} Layer;

typedef struct Network {
        Layer hidden, output;
} Network;

void network_init(Network *net, size_t input_size);
void network_train(Network *net, const f32 *input, u8 label, f32 learning_rate);
void network_free(Network *net);

#endif
