import numpy as np

data = np.load('calib_data.npz')['data']
data.resize(32*32,256,1,32)
batch_size=32

def calib_input(iter):

    calib_data = data[iter*batch_size:(iter+1)*batch_size]

    return {'input_1': calib_data}

