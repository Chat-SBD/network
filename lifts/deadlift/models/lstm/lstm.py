from tensorflow import keras

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

from lib.CONSTANTS import FRAMES, SIZE
FILTERS = 3
POOL = 20

model = keras.Sequential([
    keras.Input(shape = (FRAMES, SIZE, SIZE, 1)),
    keras.layers.TimeDistributed(
        keras.layers.Conv2D(
            filters = FILTERS,
            kernel_size = (3, 3),
            input_shape = (SIZE, SIZE, 1),
            padding = 'same'
        )
    ),
    keras.layers.TimeDistributed(
        keras.layers.MaxPooling2D(pool_size = (POOL, POOL))
    ),
    keras.layers.BatchNormalization(),
    keras.layers.Reshape((FRAMES, int((SIZE / POOL) * (SIZE / POOL) * FILTERS))),
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

model.save('lifts/deadlift/models/lstm/model')
keras.utils.plot_model(model, to_file = 'lifts/deadlift/models/lstm/lstm.png', show_shapes = True)
print('Saved model')