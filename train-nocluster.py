"""
Train a model on a batch, just normally without any parallel work.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
"""
import os
from sys import argv
from tensorflow import keras
import tensorflow as tf

from lib.network import FrameGenerator
from lib.CONSTANTS import FRAMES, SIZE

SAVEPATH = argv[1] + 'model/'
DSPATH = '/'.join(argv[1].split('/')[: 2]) + '/dataset/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Loading model...')
model = keras.models.load_model(SAVEPATH)

print('Building datasets...')
outsig = (
    tf.TensorSpec(shape = (FRAMES, SIZE, SIZE, 1), dtype = tf.int16),
    tf.TensorSpec(shape = (), dtype = tf.int16)
)
ds_train = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'train'), output_signature = outsig)
ds_val = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'val'), output_signature = outsig)

ds_train = ds_train.batch(20)
ds_val = ds_val.batch(20)

model.fit(
    x = ds_train,
    epochs = 10,
    validation_data = ds_val
)

print('Saving...')
model.save(SAVEPATH)
print('Saved model')
