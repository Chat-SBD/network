"""
Train a model on a batch.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
    2: str. The path to the lift folder. 'lifts/squat/'
"""
import os
from sys import argv
from tensorflow import keras

from lib.Server import Server

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

server = Server(
    argv[1],
    argv[2],
    keras.optimizers.Adam(),
    keras.losses.SparseCategoricalCrossentropy(from_logits = True)
)

# we've done 4 epochs so far, going for 10 total
for epoch in range(4, 11):
    server.progress(0)
    server.train(epoch)