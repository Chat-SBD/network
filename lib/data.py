from glob import glob
import numpy as np
import cv2
import random

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

# constants
from lib.CONSTANTS import SEED, SECS, FPS

def get_vids(path, seed = SEED):
    """
    Gets an array of tuples of full relative video path and number of white lights
    [('batch/train/squat-batch1-18_2', 2), (...]

    Args:
        path: str. 'batch/train/' or 'batch/test/', whichever is being used.
        seed: int, 0-100. Random seed for video shuffling.
    """
    videos = glob(path + '*.mp4')

    random.seed(seed)
    random.shuffle(videos)

    #return list(zip(videos, [0 if int(video.split('_')[1].split('.')[0]) < 2 else 1 for video in videos]))
    return list(zip(videos, [int(video.split('_')[1].split('.')[0]) for video in videos]))

def get_frames(path, secs = SECS, fps = FPS):
    """
    Return a numpy array of the frames from a video file.
    An array (frames) of arrays (pixels) of arrays (color channel values) (I think).

    Args:
        path: str. The relative path to the video file.
        secs: int. The number of seconds in the video to read in.
        fps: int. Frames per seconds to read in.
    """
    video = cv2.VideoCapture(path)
    frames = [None] * secs * fps

    frames_index = 0
    video_index = 0
    stepper = int(video.get(cv2.CAP_PROP_FPS) / fps)
    
    while frames_index < secs * fps:
        video.set(cv2.CAP_PROP_POS_FRAMES, video_index)
        ret, frame = video.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames[frames_index] = frame
        video_index += stepper
        frames_index += 1

    video.release()
    return np.expand_dims(np.array(frames), axis = -1)