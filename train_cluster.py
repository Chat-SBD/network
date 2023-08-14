"""
Train a model on a batch using tensorflow distributed training on the cluster.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
    2: str. The path to the lift folder. 'lifts/squat/'
"""
import os
import json
from sys import argv
from tensorflow import keras, distribute
from mpi4py import MPI
import numpy as np
from numpy import random
import logging

from lib.data import get_vids, get_frames, train_test_val, expand

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

world = MPI.COMM_WORLD
rank = world.Get_rank()
name = MPI.Get_processor_name()
nprocs = world.Get_size()

logging.warning(f'{name}: Setting TF_CONFIG...')

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['beowulf:12624', 'node1:12624', 'node2:12624', 'node3:12624', 'node4:12624']
    },
    'task': {
        'type': 'worker',
        'index': rank
    }
})
world.Barrier()

logging.warning(f'{name}: Starting strategy...')
strategy = distribute.MultiWorkerMirroredStrategy()
world.Barrier()

logging.warning(f'{name}: Loading model...')
with strategy.scope():
    #model = keras.models.load_model(argv[1] + 'model/')
    from lifts.squat.models.conv21d.conv21d import make
    model = make()
world.Barrier()

logging.warning(f'{name}: Loading video paths...')
data = get_vids(argv[2] + 'batch/')
x = np.array([path for path, lights in data])
y = np.array([lights for path, lights in data])
logging.warning(f'{name}: {len(x)} paths loaded')

logging.warning(f'{name}: Expanding...')
x, y = expand(x, y)
logging.warning(f'{name}: Expanded to {len(x)} paths')

logging.warning(f'{name}: Shuffling...')
random.seed(42)
random.shuffle(x)
random.seed(42)
random.shuffle(y)

logging.warning(f'{name}: Splitting...')
x_train, y_train, x_test, y_test, x_val, y_val = train_test_val(x, y)

logging.warning(f'{name}: Loading video data...')
videos = [get_frames(path) for path in x]
videos_paths = dict(zip(x, videos))

x_train = np.array([videos_paths[path] for path in x_train])
x_test = np.array([videos_paths[path] for path in x_test])
x_val = np.array([videos_paths[path] for path in x_val])
world.Barrier()

model.fit(
    x = x_train,
    y = y_train,
    epochs = 10,
    validation_data = (x_val, y_val)
)

if rank == 0:
    logging.warning(f'{name}: Evaluating...')
    model.evaluate(
        x = x_test,
        y = y_test
    )

    model.save(argv[1] + 'model/')
    logging.warning(f'{name}: Saved model')