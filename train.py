"""
Train a model on a batch, just normally without any parallel work

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
    2: str. The path to the lift folder. 'lifts/squat/'
"""
import os
from sys import argv
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

from lib.data import get_vids, get_frames

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
TEST_SIZE = 0.3
VAL_SIZE = 0.1

model = keras.models.load_model(argv[1] + 'model/')

print('Loading dataset...')
data = get_vids(argv[2] + 'batch/')
x_data = np.array([get_frames(path) for path, lights in data])
y_data = np.array([lights for path, lights in data])

print('Splitting...')
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size = TEST_SIZE,
    shuffle = True
)

placeholder, x_val, placeholder2, y_val = train_test_split(
    x_train,
    y_train,
    test_size = VAL_SIZE,
    shuffle = False
)

print(f'Test size: {int(TEST_SIZE * 100)}% of total')
print(f'Train size: {100 - int(TEST_SIZE * 100)}% of total')
print(f'Val size: {int(VAL_SIZE * 100)}% of train')

model.fit(
    x = x_train,
    y = y_train,
    epochs = 20,
    validation_data = (x_val, y_val)
)

print('Evaluating...')
model.evaluate(
    x = x_test,
    y = y_test
)

model.save(argv[1] + 'model/')
print('Saved model')