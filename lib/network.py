import tensorflow as tf

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

# constants
from lib.CONSTANTS import FRAMES

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
            tf.TensorSpec(shape = (FRAMES, 640, 640, 1), dtype = tf.int16),
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