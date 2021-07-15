#include "custom_layers.h"
#include "weights.h"

void custom_kernel_sw(dtype *data_in, dtype* data_out)
{
	dtype sum;
	dtype gap_out[GAP_SHAPE], fc_out[OUTPUT_SIZE];
	int i, j;

	for (i = 0; i < GAP_SHAPE; i++)
		{

			sum = 0;
			GAP_L2: for (int j = 0; j < WINDOW_SIZE; j++)
			{
				sum += data_in[j];
			}
			gap_out[i] = sum / WINDOW_SIZE;

		}
	dtype acc, mul, reg;

	for (i = 0; i < OUTPUT_SIZE; i++)
	{
		acc = 0;
		for (j = 0; j < GAP_SHAPE; j++)
		{
			reg = gap_out[j];
			mul = reg * WEIGHTS[j][i];
			acc += mul;
		}

		acc += BIAS[i];
		fc_out[i] = acc;
	}

	dtype tmp;
	for (i = 0 ; i < OUTPUT_SIZE; i++)
	{
		tmp = fc_out[i];

		if (tmp >= 0 && tmp < 3.4)
			data_out[i] = ((((dtype(0.75)) * tmp) / (dtype(1) + (tmp / dtype(2)))) + dtype(1)) ;

		else if (tmp >= 3.4 && tmp < 6)
			data_out[i] = (dtype(1.935409070603099) + (dtype(0.0458812946797165) * ((tmp / dtype(2)) - dtype(1.7))));

		else if (tmp > 6)
			data_out[i] = dtype(1.99505475368673);

		else if (tmp <= 0 && tmp > -3.4)
			data_out[i] = (dtype(-1) * ((dtype(-1.5/2) * tmp) / (dtype(1) + ((dtype(-1) * tmp)  / dtype(2)))) + dtype(1));

		else if (tmp <= -3.4 && tmp > -6)
			data_out[i] = (dtype(1) - (dtype(0.935409070603099) + (dtype(0.0458812946797165) * (((dtype(-1)*tmp) / dtype(2)) - dtype(1.7)))));

		else if (tmp <= -6)
			data_out[i] = dtype(0.00494524631327) ;

		data_out[i] /= 2;
	}
}

void init_array(dtype* data_in)
{
	for (int i = 0; i < WINDOW_SIZE; i++)
		for (int j = 0; j < GAP_SHAPE; j++)
			data_in[i][j] = (i+j) / 100;
}
int main(void)
{

	dtype data_in[WINDOW_SIZE][GAP_SHAPE];
	dtype data_out_hw[OUTPUT_SIZE], data_out_sw[OUTPUT_SIZE];
	int err = 0;

	init_array(data_in);
	custom_kernel_sw(data_in, data_out_sw)
	custom_layers_top(data_in, data_out_hw);

	for (int i = 0; i < OUTPUT_SIZE; i++)
		if (data_out_sw[i] != data_out_hw[i])
			err ++;

	return err;
}
