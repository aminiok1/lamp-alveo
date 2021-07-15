
from keras.datasets import mnist
from keras.utils import to_categorical
from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys
import time

from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator
import scipy.io as sio

# np.set_printoptions(threshold=sys.maxsize)

"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def CPUCalcSoftmax(data, size):

    for j in range(size):
        datanp = np.asarray(data[j])
        datanp.reshape(10)

        sum = 0.0
        
        result = [0 for i in range(10)]
    
        for i in range(10):
            result[i] = math.exp(datanp[i])
            sum += result[i]
        for i in range(10):
            result[i] /= sum
        
        print(result.index(max(result)))


def runFC(runner, data):
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])
    output_ndim = tuple(outputTensors[0].dims)

    runSize = input_ndim[0]

    sig_out = []

    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

    for i in range(0, 128, runSize):#((len(data)//runSize) + 1):
        
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = data[i % len(data)].reshape(input_ndim[1:])

    #print("imageRun shape = {}, input_ndim = {}, pre_output_size = {}, output_ndim = {}, runSize = {}".format(imageRun.shape, input_ndim, pre_output_size, output_ndim, runSize))

        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
    
        outDataNp = np.asarray(outputData)

        res = 1/(1 + np.exp(-outDataNp))
        
        sig_out.append(res)

    #return runSize, sig_out

def runLamp(runner1, runner2, runner3,  data, result, idx):

    inputTensors1 = runner1.get_input_tensors()
    outputTensors1 = runner1.get_output_tensors()
    
    input_ndim1 = tuple(inputTensors1[0].dims)
    pre_output_size1 = int(outputTensors1[0].get_data_size() / input_ndim1[0])
    output_ndim1 = tuple(outputTensors1[0].dims)

    runSize1 = input_ndim1[0]
    

    inputTensors2 = runner2.get_input_tensors()
    outputTensors2 = runner2.get_output_tensors()
    
    input_ndim2 = tuple(inputTensors2[0].dims)
    pre_output_size2 = int(outputTensors2[0].get_data_size() / input_ndim2[0])
    output_ndim2 = tuple(outputTensors2[0].dims)

    runSize2 = input_ndim2[0]
    

    inputTensors3 = runner3.get_input_tensors()
    outputTensors3 = runner3.get_output_tensors()
    
    input_ndim3 = tuple(inputTensors3[0].dims)
    pre_output_size3 = int(outputTensors3[0].get_data_size() / input_ndim3[0])
    output_ndim3 = tuple(outputTensors3[0].dims)

    runSize3 = input_ndim3[0]
    
    print("run sizes = {} {} {}".format(runSize1, runSize2, runSize3))
    
    #gap_output = []
    
    #print("runSize = {}".format(runSize))
    #print("valid_gen size = {}".format(len(valid_gen)))

    #for w in range(1):#(len(valid_gen)):

    #print("X_valid size ={}".format(len(X_valid)))
  
    inputData1 = [np.empty(input_ndim1, dtype=np.float32, order="C")]
    outputData1 = [np.empty(output_ndim1, dtype=np.float32, order="C")]
    
    inputData2 = [np.empty(input_ndim2, dtype=np.float32, order="C")]
    outputData2 = [np.empty(output_ndim2, dtype=np.float32, order="C")]
    
    inputData3 = [np.empty(input_ndim3, dtype=np.float32, order="C")]
    outputData3 = [np.empty(output_ndim3, dtype=np.float32, order="C")]


    #for i in range(0, len(data), runSize):
    for i in range(1):
         
        for j in range(runSize1):
            imageRun1 = inputData1[0]
            imageRun1[j, ...] = data[i%len(data)].reshape(input_ndim1[1:])

        start = time.time()    
        job_id = runner1.execute_async(inputData1, outputData1)
        runner1.wait(job_id)
        end = time.time()
         
         
        for j in range(runSize2):
            imageRun2 = inputData2[0]
            imageRun2[j, ...] = outputData1[0][i%len(data)].reshape(input_ndim2[1:])

        start2 = time.time()    
        job_id = runner2.execute_async(inputData2, outputData2)
        runner2.wait(job_id)
        end2 = time.time()

        for j in range(runSize3):
            imageRun3 = inputData3[0]
            imageRun3[j, ...] = outputData2[0][i%len(data)].reshape(input_ndim3[1:])

        start3 = time.time()    
        job_id = runner3.execute_async(inputData3, outputData3)
        runner3.wait(job_id)
        end3 = time.time()

    print("timeeee = {}".format((end - start) + (end2 - start2) + (end3 - start3)))


# Matrix Profile configs
matrix_profile_window = 256
sample_rate = 20
lookbehind_seconds = 0
lookahead_seconds = 0
subsequence_stride = 256
lookbehind = sample_rate * lookbehind_seconds
num_outputs = 256
lookahead = sample_rate * lookahead_seconds
forward_sequences = lookahead + num_outputs
subsequences_per_input = lookbehind + num_outputs + lookahead
channel_stride = 8
n_input_series = 1
subsequences_per_input = subsequences_per_input // channel_stride
high_weight = 1
low_thresh = -1
high_thresh = 1
batch_size = 128

# Read the input data and generate the corresponding time series
all_data = sio.loadmat('insect_no_classification.mat')

mp_val = np.array(all_data['mp_test'])
ts_val = np.array(all_data['ts_test'])

valid_gen = MPTimeseriesGenerator(data=ts_val,targets= mp_val, num_input_timeseries=1, internal_stride=8, num_outputs=256,lookahead=forward_sequences, lookbehind=lookbehind, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, length=256, mp_window=256, stride=num_outputs, batch_size=batch_size)

# Extract x_valid, y_valid
threadnum = int(sys.argv[1])
X_valid = []

for i in range(threadnum):
    data = valid_gen[i]
    X, Y_valid = data
    X = np.float32(X)

    X_valid.append(X)


g1 = xir.Graph.deserialize("lamp_pt1.xmodel")
g2 = xir.Graph.deserialize("lamp_pt2.xmodel")
g3 = xir.Graph.deserialize("lamp_pt3.xmodel")

subgraphs1 = get_child_subgraph_dpu(g1)
subgraphs2 = get_child_subgraph_dpu(g2)
subgraphs3 = get_child_subgraph_dpu(g3)

# Threading
all_dpu_runners1 = []
all_dpu_runners2 = []
all_dpu_runners3 = []

all_dpu_runners_fc = []

threadAll = []
threadAllFc = []

results = [None] * threadnum

for i in range(threadnum):
    all_dpu_runners1.append(vart.Runner.create_runner(subgraphs1[0], "run"))
    all_dpu_runners2.append(vart.Runner.create_runner(subgraphs2[0], "run"))
    all_dpu_runners3.append(vart.Runner.create_runner(subgraphs3[0], "run"))

for i in range(threadnum):
    t1 = threading.Thread(target=runLamp, args=(all_dpu_runners1[i], all_dpu_runners2[i], all_dpu_runners3[i], X_valid[i], results, i))
    threadAll.append(t1)

time_start = time.time()

for x in threadAll:
    x.start()

for x in threadAll:
    x.join()

time_end = time.time()

print("total time = {}".format(time_end - time_start))
exit(0)
for i in range(threadnum):
    all_dpu_runners_fc.append(vart.Runner.create_runner(subgraphs[1], "run"))
    
for i in range(threadnum):
    t1 = threading.Thread(target=runFC, args=(all_dpu_runners_fc[i], results[i]))
    threadAllFc.append(t1)

for x in threadAllFc:
    x.start()

for x in threadAllFc:
    x.join()


#runner = vart.Runner.create_runner(subgraphs[0], "run")
#runSize, outData = runLamp(runner, X_valid)

'''
outData.resize(len(outData) * runSize, 192)

runner2 = vart.Runner.create_runner(subgraphs[1], "run")
fcRunSize, res = runFC(runner2, outData) 


#for i in range(len(valid_gen)):
#    data = valid_gen[i]
#    X_valid, Y_valid = data

#    y.append(Y_valid)

#y = np.asarray(y)
#y.resize(len(valid_gen) * 128, 256)

data_ctr = -1
yindx = 0
final_res = 0

for j in range(len(res)):
    r = np.asarray(res[j])
    r.resize(fcRunSize, 256)

    for i in range(fcRunSize):
        if ((j * fcRunSize + i) % 128 == 0):
            data_ctr += 1
            yindx = 0
            data = valid_gen[data_ctr]
            X_valid, Y_valid = data

        final = (np.mean(np.abs((r[i] - Y_valid[yindx])/r[i])) * 100)
        print(final)
        final_res += final
        yindx += 1



print("final = {}".format(final_res))
'''
