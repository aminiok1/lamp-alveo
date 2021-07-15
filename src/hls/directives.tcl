############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_interface -mode m_axi -depth 19200 -offset slave -bundle in_aximm "custom_layers_top" data_in
set_directive_interface -mode m_axi -depth 256 -offset slave -bundle out_aximm "custom_layers_top" data_out
set_directive_interface -mode s_axilite -bundle control "custom_layers_top"
set_directive_dataflow "custom_layers_top"
set_directive_pipeline "sigmoid/SIG_L1"
set_directive_pipeline "global_average_pool/GAP_L1"
set_directive_pipeline "dense/DENSE_L0"
set_directive_pipeline "dense/DENSE_L5"
set_directive_unroll -factor 4 "dense/DENSE_L3"
set_directive_unroll -factor 4 "dense/DENSE_L4"
set_directive_array_partition -type block -factor 4 -dim 1 "dense" intermediate_vals
set_directive_array_partition -type block -factor 4 -dim 1 "custom_layers_top" gap_out
set_directive_pipeline "dense/DENSE_L1"
