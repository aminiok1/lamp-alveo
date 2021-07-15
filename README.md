
LAMP-ALVEO
=======================
This repository is a walk-through for running LAMP on Xilinx® Alveo™ U280 FPGA card. The Ultra96-V2 implementation and guide can be accessed from [this link](https://github.com/aminiok1/fccm-lamp). The instructions require [Docker](https://github.com/Xilinx/Vitis-AI/blob/v1.0/doc/install_docker/README.md), [Vitis-AI 1.3](https://github.com/Xilinx/Vitis-AI),  and [Vitis 2019.2](https://www.xilinx.com/html_docs/xilinx2019_2/vitis_doc/Chunk1674708719.html),  installed on the system. Read more about Matrix Profile and LAMP at the [Matrix Profile Homepage](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html).

# Folder Structure
    .
    ├── data                # Sample data for training and quantization
    ├── models              # Pre-trained models, weight values for custom kernel
    ├── scripts             # Scripts for generating the compiled model for DPU and evalution
    ├── src
    │   ├── hls				# Custom kernel HLS files
    │   └── host			# Source code for host CPU
    ├── vitis               # Configuration files for building the custom kernel
    ├── LICENSE
    └── README.md
# Build Instructions
In order to execute LAMP inference on Alveo card, we start by running Keras quantization aware training method on the pre-trained LAMP model. The quantization aware model is then compiled using Vitis-AI. We also synthesize and package the custom kernel using Vitis HLS and generate a device binary file using Vitis compiler. Finally, we run the model by executing the host code written in Python.
<H2>Quantization Aware Training</H2>
First you need to install the following packages.
<H4>Setup</H4>

```
$ pip install -q tensorflow
$ pip install -q keras
$ pip install -q tensorflow-model-optimization
```
After that, you can use the provided script to generate the quantization aware model from a pre-trained model.
```
$ python qa_train.py [model] [data]
```
Where model is the pre-trained model in <code>h5</code> format and data is the input time series training data. The script will generate a <code>lamp_qa.h5</code> model that we use in the next step.

<H2>Compiling the Model</H2>
Launch the docker tools from Vitis-AI directory

```shell
$ sh -x docker_run.sh xilinx/vitis-ai:1.3.411
$ conda activate vitis-ai-tensorflow
```
 <H3>1. Freezing Tensorflow graph </H3>
The Vitis-AI flow requires  a frozen model for quantization and optimization steps. A frozen model contains information about the graph and checkpoint variables, saving these hyperparameters as constants within the graph structure. This allows fusing some of the layers together for deployment on DPU. We can generate a binary protobuf (.pb) file by running the <code>freeze_graph.py</code> script
  
  ```shell
 $ python freeze_graph.py [input_model]
  ```
  where <code>input_model</code> is the quantization aware model generated in the previous step
  
  <H3>2. Quantization </H3>
  
  We will quantize the weights/biases and activations of the model to improve the performance of the model inference on FPGA. Currently, Xilinx DPU only supports 8 bit models, so we quantize everything to 8 bits.
```shell
$ vai_q_tensorflow quantize 
                 --input_frozen_graph frozen_graph.pb 
                 --input_fn input_func.calib_input
                 --output_dir quantized 
                 --input_nodes input_1 
                 --output_nodes reshape_1/Reshape 
                 --input_shapes ?,256,1,32 
                 --calib_iter 32
```
<code>frozen_graph.pb</code> is the frozen model generated in the previous step, <code>input_func</code> is the python file that generates the input data for quantizer (since there is no backpropagation step here, the unlabeled dataset is sufficient), and <code>calib_iter</code> is the number of iterations for calibrating the activations, we noticed that values larger than 32 do not increase the quantizer accuracy by a lot.
<H3> 3. Evaluation</H3>
We will test the accuracy of the generate quantized model before deploying it to the FPGA. 

```shell
$ python evaluate.py
```
<code>evaluate.py</code> reads in the Tensorflow frozen binary graph, runs the inference and reports the least squared  error by comparing the model output with the labels (matrix profile values). 
<H3>4. Compilation</H3>
 Next, we will compile the model for the target hardware
 
 ```
$ vai_c_tensorflow --frozen_pb quantized/quantize_eval_model.pb 
                  --arch /opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json 
                  --output_dir . 
                  --net_name lamp
 ```
This will generate the compiled model <code>lamp.xmodel</code> for the High Throughput DPU that we will use in our host code. In order to compile the model for the Low Latency DPU, replace <code>DPUCAHX8H</code> with <code>DPUCAHX8L</code> in the above command.

<H2>HLS Kernel</H2>
To generate the IP file for the custom kernel, in hls directory run 

  ```shell
 $ vivado_hls -f script.tcl
  ```
 The sigmoid implementation can be configured by either choosing <code>SIGMOID_ULTRA_FAST</code>  or <code>SIGMOID_EXP_512</code> in <code>defines.h</code> file. Other <code>weights.cpp</code> files from models directory can be replaced with the original file to evaluate different benchmarks.
Next, we use the generated <code>.xo</code> IP file to build the final xclbin by running the following command

```
$ v++ --link                               \
--target hw                          \
--platform xilinx_u280_xdma_201920_1 \
--config connectivity.cfg          \
--output custom_kernel.xclbin   \
custom_layers.xo
```
The config file can be found in the vitis directory. We will load the <code>custom_kernel.xcl</code> in the host code.

<H2>Running inference</H2>
<H4>Setup</H4>

```
$ wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.3.0.tar.gz
$ tar -xzvf alveo_xclbin-1.3.0.tar.gz

# for High Throughput DPU
$ cd alveo_xclbin-1.3.1/U280/14E300M/

# for Low Latency DPU
$ cd alveo_xclbin-1.3.1//U280-V3ME/2E300M/

$ sudo cp dpu.xclbin /usr/lib/hbm_address_assignment.txt
```

After that you are able to run the  host code by running the following (for the Low  Latency DPU, make sure that [xbutler](https://github.com/Xilinx/Vitis-AI/tree/1.2/alveo/packages) is running):

```
$ python host.py [thread_no]
```
