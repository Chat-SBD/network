import tensorflow as tf

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