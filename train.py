"""
Train a model on a batch, just normally without any parallel work

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
from time import sleep

from lib.network import FrameGenerator

SAVEPATH = argv[1] + 'model/'
DSPATH = '/'.join(argv[1].split('/')[: 2]) + '/dataset/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

world = MPI.COMM_WORLD
rank = world.Get_rank()
name = MPI.Get_processor_name()
nprocs = world.Get_size()

logging.warning(f'{name}:Setting TF_CONFIG...')

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['beowulf:12624', 'node1:12624', 'node2:12624', 'node3:12624', 'node4:12624']
    },
    'task': {
        'type': 'worker',
        'index': rank
    }
})
sleep(15)
world.Barrier()

logging.warning(f'{name}:Starting strategy...')
strategy = distribute.MultiWorkerMirroredStrategy()
world.Barrier()

logging.warning(f'{name}:Loading model...')
with strategy.scope():
    model = keras.models.load_model(SAVEPATH)
world.Barrier()

logging.warning(f'{name}:Building datasets...')
outsig = (
    tf.TensorSpec(shape = (None, None, None, 1), dtype = tf.int16),
    tf.TensorSpec(shape = (), dtype = tf.int16)
)
ds_train = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'train'), output_signature = outsig)
ds_test = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'test'), output_signature = outsig)
ds_val = tf.data.Dataset.from_generator(FrameGenerator(DSPATH, 'val'), output_signature = outsig)
world.Barrier()

model.fit(
    x = ds_train,
    epochs = 2,
    validation_data = ds_val
)
world.Barrier()

logging.warning(f'{name}:Evaluating...')
model.evaluate(
    x = ds_test
)
world.Barrier()

logging.warning(f'{name}:Saving...')
if rank == 0:
    model.save(SAVEPATH)
else:
    model.save('/tmp/model')
logging.warning(f'{name}:Saved model')
world.Barrier()