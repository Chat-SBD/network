import tensorflow as tf
from tensorflow import keras
from numpy import random

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

from lib.CONSTANTS import FPS, SECS, FRAMES, SIZE, TEST_SIZE, VAL_SIZE
from lib.data import get_vids, expand, train_test_val, get_frames, compress

class FrameGenerator:
    """
    Class to yield video data as needed to prevent unecessary memory usage.

    Args:
        path: str. Path to the directory that the videos are in, 'lifts/squat/dataset/'.
        portion: str. 'train'|'test'|'val' for each kind of dataset.
    """
    def __init__(self, path, portion):
        paths_lights = get_vids(path)
        x = [path for path, lights in paths_lights]
        y = [lights for path, lights in paths_lights]

        x, y = compress(x, y)

        random.seed(42)
        random.shuffle(x)
        random.seed(42)
        random.shuffle(y)

        x_train, y_train, x_test, y_test, x_val, y_val = train_test_val(x, y)

        if portion == 'train':
            self.x = x_train
            self.y = y_train
        
        elif portion == 'test':
            self.x = x_test
            self.y = y_test
        
        elif portion == 'val':
            self.x = x_val
            self.y = y_val
    
    def __call__(self):
        for x, y in zip(self.x, self.y):
            yield get_frames(x), y
