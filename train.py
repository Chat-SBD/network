"""
Train a model on a batch.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
    2: str. The path to the lift folder. 'lifts/squat/'
"""
from sys import argv
from tensorflow import keras

from lib.Server import Server

server = Server(
    argv[1],
    argv[2],
    keras.optimizers.Adam(),
    keras.losses.SparseCategoricalCrossentropy()
)

server.train(secs = 28, fps = 24, seed = 42)