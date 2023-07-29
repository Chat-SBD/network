from tensorflow import keras

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

from lib.data import get_frames
from lib.network import dataset
from lib.CONSTANTS import FRAMES, SIZE

vid_shape = (FRAMES, SIZE, SIZE, 1)

model = keras.Sequential([
    keras.Input(shape = vid_shape),
    keras.layers.Reshape((96, SIZE * SIZE), input_shape = vid_shape),
    keras.layers.LSTM(
        units = 1000,
        time_major = False
    ),
    keras.layers.Dense(units = 4, activation = 'softmax')
])
print('Created model...')

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(),
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
)
print('Compiled model...')

model.save('lifts/squat/models/lstm/model')
keras.utils.plot_model(model, to_file = 'lifts/squat/models/lstm/lstm.png', show_shapes = True)
print('Saved model')