"""
Train a model on a batch, just normally without any parallel work

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
    2: str. The path to the lift folder. 'lifts/squat/'
"""
import os
from sys import argv
from tensorflow import keras
import numpy as np
from numpy import random

from lib.data import get_vids, get_frames, train_test_val, expand

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.models.load_model(argv[1] + 'model/')

print('Loading video paths...')
data = get_vids(argv[2] + 'batch/')
x = np.array([path for path, lights in data])
y = np.array([lights for path, lights in data])
print(f'{len(x)} paths loaded')

print('Expanding...')
x, y = expand(x, y)
print(f'Expanded to {len(x)} paths')

print('Shuffling...')
random.seed(42)
random.shuffle(x)
random.seed(42)
random.shuffle(y)

print('Splitting...')
x_train, y_train, x_test, y_test, x_val, y_val = train_test_val(x, y)

print('Loading video data...')
videos = [get_frames(path) for path in x]
videos_paths = dict(zip(x, videos))

x_train = np.array([videos_paths[path] for path in x_train])
x_test = np.array([videos_paths[path] for path in x_test])
x_val = np.array([videos_paths[path] for path in x_val])

model.fit(
    x = x_train,
    y = y_train,
    epochs = 10,
    validation_data = (x_val, y_val)
)

print('Evaluating...')
model.evaluate(
    x = x_test,
    y = y_test
)

model.save(argv[1] + 'model/')
print('Saved model')