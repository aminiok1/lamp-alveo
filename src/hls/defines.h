#ifndef DEFINES_H
#define DEFINES_H

#include "ap_fixed.h"

#define SIGMOID_ULTRA_FAST
//#define SIGMOID_EXP_512

typedef ap_fixed<8,4,AP_TRN_ZERO > dtype;
typedef ap_uint<32> axitype;

#define WINDOW_SIZE		100
#define WINDOW_SIZE_FIXED	25 //(WINDOW_SIZE >>2)
#define GAP_SHAPE		192
#define OUTPUT_SIZE		256
#define OUTPUT_SIZE_FIXED		256 //(OUTPUT_SIZE >> 2)
#define ITERATOR_BITS	9
#define TILE_SIZE		4

#endif
