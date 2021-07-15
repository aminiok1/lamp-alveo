#ifndef CUSTOM_LAYERS_H
#define CUSTOM_LAYERS_H

#include "defines.h"

int custom_layers_top(const dtype *data_in, dtype *data_out);
void global_average_pool(const axitype *data_in, hls::stream<dtype> &data_out);
void dense(hls::stream<dtype> &data_in, hls::stream<dtype> &data_out);
void sigmoid(hls::stream<dtype> &data_in, axitype* data_out);

#endif
