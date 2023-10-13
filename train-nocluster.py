"""
Train a model on a batch, just normally without any parallel work.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
"""
import os
import json
from sys import argv
from tensorflow import keras
import tensorflow as tf
import tensorflowjs as tfjs

from lib.network import FrameGenerator
from lib.CONSTANTS import FRAMES, SIZE

SAVEPATH_H5 = argv[1] + 'model.h5'
SAVEPATH_JS = argv[1] + 'model/'
DSPATH = '/'.join(argv[1].split('/')[: 2]) + '/dataset/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Loading model...')
model = keras.models.load_model(SAVEPATH_H5)

print('Building datasets...')
outsig = (
    tf.TensorSpec(shape = (FRAMES, SIZE, SIZE, 1), dtype = tf.int16),
    tf.TensorSpec(shape = (), dtype = tf.int16)
)
ds_train = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'train'), output_signature = outsig)
ds_test = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'test'), output_signature = outsig)
ds_val = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'val'), output_signature = outsig)

ds_train = ds_train.batch(100)
ds_test = ds_test.batch(100)
ds_val = ds_val.batch(100)

model.fit(
    x = ds_train,
    epochs = 1,
    verbose = 1,
    validation_data = ds_val
)

print('Saving...')
model.save(SAVEPATH_H5)
tfjs.converters.save_keras_model(model, SAVEPATH_JS)
print('Saved model')