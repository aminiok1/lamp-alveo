import sys
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import scipy.io as sio
import numpy as np

from tensorflow import keras
from TimeSeriesGeneratorAdapt import MPTimeseriesGenerator


# Generates the trainig time series data for the model
def prepare_input():
    all_data = sio.loadmat(sys.argv[2])

    # Load training data
    mp = np.array(all_data['mp_train'])
    ts = np.array(all_data['ts_train'])
	
    # Load validation data
    mp_val = np.array(all_data['mp_val'])
    ts_val = np.array(all_data['ts_val'])
	
    # Neural network and time series parameters
    matrix_profile_window = 256
    input_width = matrix_profile_window
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

    # Generators for model input
    valid_gen = MPTimeseriesGenerator(ts_val, mp_val, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs,
                                    lookahead=forward_sequences, lookbehind=lookbehind, important_upper_threshold=high_thresh, important_lower_threshold=low_thresh,
                                    important_weight=high_weight, length=input_width, mp_window=matrix_profile_window, stride=num_outputs, batch_size=batch_size)
    train_gen = MPTimeseriesGenerator(ts, mp, num_input_timeseries=n_input_series, internal_stride=channel_stride, num_outputs=num_outputs, 
                                    lookahead=forward_sequences, lookbehind=lookbehind, length=input_width, mp_window=matrix_profile_window, stride=subsequence_stride, 
                                    important_upper_threshold=high_thresh, important_lower_threshold=low_thresh, important_weight=high_weight, batch_size=batch_size, shuffle=True, merge_points=None)
	
    return valid_gen, train_gen
	
 
# Load the pre-trained model
loaded_model = keras.models.load_model(sys.argv[1])

quantize_model = tfmot.quantization.keras.quantize_model

# Call the quantization aware api
q_aware_model = quantize_model(loaded_model)
 
# recompile the model
q_aware_model.compile(optimizer='adam',
              loss='mse')

q_aware_model.summary()

# Train the model against baseline
nb_epochs = 1
valid_gen, train_gen = prepare_input()
q_aware_model.fit_generator(train_gen, workers=6, use_multiprocessing=True, validation_data=valid_gen, shuffle=True, epochs=nb_epochs,verbose=1, initial_epoch=0)

q_aware_model.save("lamp_qa.h5")
