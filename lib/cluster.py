import cv2
from glob import glob
import numpy as np
import random
from mpi4py import MPI
import tensorflow as tf

def get_vids(seed, path, start = 0):
    """
    Gets an array of tuples of full relative video path and number of white lights
    [('batch/train/squat-batch1-18_2', 2), (...]

    Args:
        seed: int, 0-100. Random seed for video shuffling.
        path: str. 'batch/train/' or 'batch/test/', whichever is being used. 
        start: int = 0. Index, only return videos after this from the shuffled list.
    """
    videos = glob(path + '*.mp4')

    random.seed(seed)
    random.shuffle(videos)

    return zip(videos[start :], int(videos[start :].split('_')[1].split('.')[0]))

def get_frames(path, secs, fps):
    """
    Return a numpy array of the frames from a video file.
    An array (frames) of arrays (pixels) of arrays (color channel values) (I think).

    Args:
        path: str. The relative path to the video file.
        secs: int. The number of seconds in the video to read in.
        fps: int. Frames per seconds to read in.
    """
    video = cv2.VideoCapture(path)
    frames = []

    index = 0
    stepper = int(video.get(cv2.CAP_PROP_FPS) / fps)
    
    while len(frames) < secs * fps:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video.read()
        
        frames.append(frame)
        index += stepper

    video.release()
    return np.array(frames)

class Server:
    def __init__(self, path):
        """
        A class to manage cluster nodes, serve videos to them, and average/apply their gradients.

        Args:
            path: str. The relative path to a file to write training progress to. 'batch/progress.txt'
        """
        self.path = path
        self.world = MPI.COMM_WORLD
        self.rank = self.world.Get_rank()
    
    def progress(self, index):
        """
        Write a number to a file.

        Args:
            index: any. The int, usually, to overwrite to the file.
        """
        with open(self.path, 'w') as progfile:
            progfile.write(str(index))

output_sig = (
    tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.int16),
    tf.TensorSpec(shape = (), dtype = tf.int8)
)

train_ds = tf.data.Dataset.from_generator(
    VideoGen('batch/train/', 28, 1),
    output_signature = output_sig
)

frames, labels = next(iter(train_ds))
print(frames.shape)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)