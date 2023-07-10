import tensorflow as tf
import numpy as np

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

# constants
from lib.CONSTANTS import FRAMES, SIZE

class VideoGen:
    """
    Basically mocked-up generator object to serve the frames and lights from a single video
    """
    def __init__(self, frames, lights):
        self.frames = frames
        self.lights = lights
    
    def __call__(self):
        yield self.frames, self.lights

def dataset(frames, lights):
    """
    Convert data into a Dataset-yielded tuple for correct sizing.

    Args:
        frames: Numpy array.
        lights: int.
    """
    return next(iter(tf.data.Dataset.from_generator(
        VideoGen(frames, lights),
        output_signature = (
            tf.TensorSpec(shape = (FRAMES, SIZE, SIZE, 1), dtype = tf.int16),
            tf.TensorSpec(shape = (), dtype = tf.int8)
        )
    ).batch(1)))


def gradient(model, lossf, data):
    """
    Find a gradient on a model with some data.

    Args:
        model: tensorflow.keras.Model. The model to find the gradients on.
        lossf: tensorflow.keras.Loss. The loss function to use.
        data: tuple. (video frames, number of white lights).
    """
    frames, lights = data
    with tf.GradientTape() as tape:
        preds = model(frames, training = True)
        loss = lossf(lights, preds)

    return tape.gradient(loss, model.trainable_weights)

def evaluate(model, lossf, datas):
    """
    Get the loss and accuracy of a model on a set of training data.

    Args:
        model: tensorflow.keras.Model. The model to evaluate.
        lossf: tensorflow.keras.Loss. The loss function to use.
        datas: list of tuples. [(video frames, number of white lights), ...].
    """
    ndatas = 0
    sumloss = 0
    ncorrect = 0

    for frames, lights in datas:
        ndatas += 1
        sumloss += lossf(lights, model(frames)).numpy()

        # find output predictions
        preds = model(frames).numpy()[0]

        # if most favored output is correct...
        if np.where(np.max(preds)) == lights:
            ncorrect += 1

    loss = sumloss / ndatas
    acc = ncorrect / ndatas

    return loss, acc