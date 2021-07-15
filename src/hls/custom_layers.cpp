#include <stdio.h>
#include <string.h>
#include <hls_math.h>
#include <hls_stream.h>
#include "custom_layers.h"
#include "weights.h"

namespace hls {

extern "C"{
int custom_layers_top(const axitype *data_in, axitype *data_out)
{

	static hls::stream<dtype> gap_out("gap_out1");
	static hls::stream<dtype> dense_out("dense_out1");

	global_average_pool(data_in, gap_out);
	dense(gap_out, dense_out);
	sigmoid(dense_out, data_out);

	return 0;
}

}
}
void global_average_pool(const axitype *data_in, hls::stream<dtype> &data_out)
{
	axitype buffer[WINDOW_SIZE_FIXED];
	dtype sum = 0;
	ap_uint<ITERATOR_BITS> i;

	GAP_L1: for (i = 0; i < GAP_SHAPE; i++)
	{
		memcpy(buffer, (const axitype*)(data_in + i*WINDOW_SIZE_FIXED), WINDOW_SIZE_FIXED*sizeof(axitype));

		sum = 0;
		GAP_L2: for (int j = 0; j < WINDOW_SIZE_FIXED; j++)
		{
			sum += buffer[j].range(7, 0);
			sum += buffer[j].range(15, 8);
			sum += buffer[j].range(23, 16);
			sum += buffer[j].range(31, 24);
		}
		data_out << sum / WINDOW_SIZE;

	}

}

void dense(hls::stream<dtype> &data_in, hls::stream<dtype> &data_out)
{
	static dtype mul, stream_data;
	//static dtype int_res[OUTPUT_SIZE];

	ap_uint<ITERATOR_BITS> i, j;

	// normal multiplication

	/*
	DENSE_L1: for (j = 0; j < GAP_SHAPE; j++)
	{
		data_in >> stream_data;
		DENSE_L2:for (i = 0; i < OUTPUT_SIZE; i++)
		{

			mul = stream_data * WEIGHTS[j][i];
			int_res[i] += mul;
		}
	}

	DENSE_L3: for(i = 0; i < OUTPUT_SIZE; i++)
	{
		int_res[i] += BIAS[i];
		data_out << int_res[i];
	}


	dtype acc;
	DENSE_L1: for (i = 0; i < OUTPUT_SIZE; i++)
	{
		acc = 0;
		DENSE_L2: for (j = 0; j < GAP_SHAPE; j++)
		{
			data_in >> stream_data;
			mul = stream_data * WEIGHTS[j][i];
			acc += mul;
		}

		acc += BIAS[i];
		data_out << acc;
	}
	*/

	// tiled based row-wise multiplication

	 	ap_uint<ITERATOR_BITS> bi,  ii, jj, row_idx, col_idx;

	  	dtype intermediate_vals[OUTPUT_SIZE];
		DENSE_L0: for (bi = 0; bi < OUTPUT_SIZE; bi++)
			intermediate_vals[bi] = BIAS[bi];

		DENSE_L1: for (i = 0; i < GAP_SHAPE; i+=TILE_SIZE)
			DENSE_L2: for (j = 0; j < OUTPUT_SIZE; j+=TILE_SIZE)
			{
				row_idx = i;
				col_idx = j;

				DENSE_L3: for(ii = 0; ii < TILE_SIZE; ii++)
				{
					DENSE_L4: for(jj = 0; jj < TILE_SIZE; jj++)
					{
						mul = data_in[row_idx] * WEIGHTS[row_idx][col_idx];
						intermediate_vals[col_idx] += mul;
						col_idx++;
					}

					row_idx++;
				}
			}
		DENSE_L5: for (bi = 0; bi < OUTPUT_SIZE; bi++)
					data_out[bi] = intermediate_vals[bi];


}

void sigmoid(hls::stream<dtype> &data_in, axitype* data_out)
{
	ap_uint<ITERATOR_BITS> i, j;
	dtype tmp, result[OUTPUT_SIZE];
	axitype result_buffer[OUTPUT_SIZE_FIXED];

#if def SIGMOID_EXP_512
	dtype res, res1, res2, res3, res4, res5, res6, res7, res8, res9;

	SIG_L1: for (i = 0; i < OUTPUT_SIZE; i++)
	{
		data_in >> tmp;
		tmp = tmp * dtype(-0.11);

		res1 = dtype(1) + tmp;
		res2 = (dtype(1) + (tmp/dtype(2))) * (dtype(1) + (tmp/dtype(2)));
		res3 = (dtype(1) + (tmp/dtype(3))) * (dtype(1) + (tmp/dtype(3))) * (dtype(1) + (tmp/dtype(3)));
		res4 = (dtype(1) + (tmp/dtype(4))) * (dtype(1) + (tmp/dtype(4))) * (dtype(1) + (tmp/dtype(4))) * (dtype(1) + (tmp/dtype(4)));
		res5 = (dtype(1) + (tmp/dtype(5))) * (dtype(1) + (tmp/dtype(5))) * (dtype(1) + (tmp/dtype(5))) * (dtype(1) + (tmp/dtype(5))) * (dtype(1) + (tmp/dtype(5)));
		res6 = (dtype(1) + (tmp/dtype(6))) * (dtype(1) + (tmp/dtype(6))) * (dtype(1) + (tmp/dtype(6))) * (dtype(1) + (tmp/dtype(6))) * (dtype(1) + (tmp/dtype(6))) * (dtype(1) + (tmp/dtype(6)));
		res7 = (dtype(1) + (tmp/dtype(7))) * (dtype(1) + (tmp/dtype(7))) * (dtype(1) + (tmp/dtype(7))) * (dtype(1) + (tmp/dtype(7))) * (dtype(1) + (tmp/dtype(7))) * (dtype(1) + (tmp/dtype(7))) * (dtype(1) + (tmp/dtype(7)));
		res8 = (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8))) * (dtype(1) + (tmp/dtype(8)));
		res9 = (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9))) * (dtype(1) + (tmp/dtype(9)));
		res = res1 * res2 * res3 * res4 * res5 * res6 * res7 * res8 * res9;

		result[i] = dtype(1) / (dtype(1) + res);
	}
#endif

#ifdef SIGMOID_ULTRA_FAST
	SIG_L1: for (i = 0 ; i < OUTPUT_SIZE; i++)
	{
		data_in >> tmp;

		if (tmp >= 0 && tmp < 3.4)
			result[i] = ((((dtype(0.75)) * tmp) / (dtype(1) + (tmp / dtype(2)))) + dtype(1)) ;

		else if (tmp >= 3.4 && tmp < 6)
			result[i] = (dtype(1.935409070603099) + (dtype(0.0458812946797165) * ((tmp / dtype(2)) - dtype(1.7))));

		else if (tmp > 6)
			result[i] = dtype(1.99505475368673);

		else if (tmp <= 0 && tmp > -3.4)
			result[i] = (dtype(-1) * ((dtype(-1.5/2) * tmp) / (dtype(1) + ((dtype(-1) * tmp)  / dtype(2)))) + dtype(1));

		else if (tmp <= -3.4 && tmp > -6)
			result[i] = (dtype(1) - (dtype(0.935409070603099) + (dtype(0.0458812946797165) * (((dtype(-1)*tmp) / dtype(2)) - dtype(1.7)))));

		else if (tmp <= -6)
			result[i] = dtype(0.00494524631327) ;

		result[i] /= 2;
	}
#endif

	SIG_L2: for (j = 0; j < OUTPUT_SIZE; j=j+4)
			{
				result_buffer[j].range(7,0) = result[j];
				result_buffer[j].range(15,8) = result[j+1];
				result_buffer[j].range(23,16) = result[j+2];
				result_buffer[j].range(31,24) = result[j+3];
			}

	memcpy(data_out, (const axitype*)result_buffer, OUTPUT_SIZE_FIXED*sizeof(axitype));
}

