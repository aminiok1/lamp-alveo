############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project custom_layers_bottom_up
set_top custom_layers_top
add_files ../custom_layers/weights.h
add_files ../custom_layers/weights.cpp
add_files ../custom_layers/defines.h
add_files ../custom_layers/custom_layers.h
add_files ../custom_layers/custom_layers.cpp
add_files -tb ../custom_layers/custom_layers_test.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_optimization_level 0 -vivado_phys_opt none -vivado_report_level 0 -xo D:/Research/FCCM/hls/custom_layers_bottomup/ultrafast.xo
config_schedule -effort medium -enable_dsp_full_reg -relax_ii_for_timing -verbose=0
config_sdx -target xocc
config_rtl -encoding onehot -kernel_profile=0 -module_auto_prefix -mult_keep_attribute=0 -register_reset_num 3 -reset control -reset_async=0 -reset_level high -verbose=0
config_compile -name_max_length 80 -no_signed_zeros=0 -pipeline_loops 64 -unsafe_math_optimizations=0
config_interface -clock_enable=0 -expose_global=0 -m_axi_addr64 -m_axi_offset off -register_io off -trim_dangling_port=0
set_clock_uncertainty 27%
source "./custom_layers_bottom_up/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -flow syn -rtl verilog -format ip_catalog -xo D:/Research/FCCM/hls/custom_layers_bottomup/custom_layers.xo
