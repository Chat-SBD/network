import cv2
from glob import glob
import numpy as np
import random
import tensorflow as tf

class VideoGen:
    def __init__(self, dirpath, length, fps):
        """
        dirpath: 'batch/test/' or 'batch/train/'
        frames: CURRENTLY NOT IN USE (will be possibly needed after first training set)
        fps: desired fps of videos post processing (irrelevant after first training set)
        """
        self.dirpath = dirpath
        self.length = length
        self.fps = fps
    
    def yield_vids(self):
        """
        Yields full video path, number of white lights
        """
        vidlist = glob(self.dirpath + '*.mp4')
        random.shuffle(vidlist)
        for path in vidlist:
            yield path, int(path.split('_')[1].split('.')[0])
    
    def get_frames(self, path):
        video = cv2.VideoCapture(path)
        frames = []
        index = 0
        # how much to add to index each loop
        # AFTER FIRST ROUND OF TRAINING, STEPPER SHOULD BE A CONSTANT AMOUNT
        # FOR REST OF BATCHES
        stepper = int(video.get(cv2.CAP_PROP_FPS) / self.fps)
        while len(frames) < self.length * self.fps:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            frames.append(frame)
            index += stepper
        video.release()

        return np.array(frames)
    
    def __call__(self):
        """
        Yields a numpy array of frames, number of white lights
        """
        for path, number in list(self.yield_vids()):
            yield self.get_frames(path), number

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