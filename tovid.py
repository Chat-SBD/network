from time import sleep
from imageio import mimsave

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

from lib.data import get_frames

secs = 12
fps = 12

frames = get_frames('lifts/squat/models/squat-sample.mp4', secs = secs, fps = fps)
print(frames.shape)
mimsave('video-test.mp4', frames, fps = fps)