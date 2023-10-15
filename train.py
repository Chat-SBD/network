"""
Train a model on a batch, just normally without any parallel work.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
"""
import os
import json
from sys import argv
from tensorflow import keras, distribute
import tensorflow as tf
from mpi4py import MPI
import logging

from lib.network import FrameGenerator
from lib.CONSTANTS import FRAMES, SIZE

SAVEPATH_H5 = argv[1] + 'model.h5'
DSPATH = '/'.join(argv[1].split('/')[: 2]) + '/dataset/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

world = MPI.COMM_WORLD
rank = world.Get_rank()
name = MPI.Get_processor_name()
nprocs = world.Get_size()

logging.warning(f'{name}:Setting TF_CONFIG...')

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['beowulf:8080', 'node1:8080', 'node2:8080', 'node3:8080', 'node4:8080']
    },
    'task': {
        'type': 'worker',
        'index': rank
    }
})
world.Barrier()

logging.warning(f'{name}:Starting strategy...')
strategy = distribute.MultiWorkerMirroredStrategy()
world.Barrier()

logging.warning(f'{name}:Loading model...')
with strategy.scope():
    model = keras.models.load_model(SAVEPATH_H5)
world.Barrier()

logging.warning(f'{name}:Building datasets...')
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
world.Barrier()

model.fit(
    x = ds_train,
    epochs = 20,
    verbose = 0 if rank != 0 else 1,
    validation_data = ds_val
)
world.Barrier()

logging.warning(f'{name}:Saving...')
if rank == 0:
    model.save(SAVEPATH_H5)
else:
    model.save('/tmp/model.h5')
logging.warning(f'{name}:Saved model')

# then run `tensorflowjs_converter --input_format keras lifts/squat/models/lstm/model.h5 lifts/squat/models/lstm/modeljs/`
# you might have to mess around with custom/non-custom tf installs to use tensorflowjs_converter