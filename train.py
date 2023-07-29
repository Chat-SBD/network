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

from lib.data import get_vids, get_frames

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.models.load_model(argv[1] + 'model/')

print('Loading training set...')
train = get_vids(argv[2] + 'batch/train/')
x_train = np.array([get_frames(path) for path, lights in train])
y_train = np.array([lights for path, lights in train])

print('Loading testing set...')
test = get_vids(argv[2] + 'batch/test/')
x_test = np.array([get_frames(path) for path, lights in test])
y_test = np.array([lights for path, lights in test])

model.fit(
    x = x_test,
    y = y_test,
    batch_size = 20,
    epochs = 50,
    validation_data = (x_test, y_test)
)

model.save(argv[1] + 'model/')
print('Saved model')