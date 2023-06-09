import cv2
import imageio
import numpy as np
import random

VIDDIR = 'UCF101/'
VIDPATH = 'batch/train/clip48_2.mp4'

src = cv2.VideoCapture(VIDPATH)
result = [frame for ret, frame in iter(src.read, (False, None))]
src.release()

result = np.array(result)[..., [2, 1, 0]]

class VideoGen:
    def __init__(self, dirpath, frames, fps):
        """
        dirpath: 'batch/test/' or 'batch/train/'
        frames: CURRENTLY NOT IN USE (will be possibly needed after first training set)
        fps: desired fps of videos post processing (irrelevant after first training set)
        """
        self.dirpath = dirpath
        self.frames = frames
        self.fps = fps
    
    def yield_vids(self):
        """
        Yields full video path, number of white lights
        """
        for path in random.shuffle(list(self.dirpath.glob('*.mp4'))):
            yield path, int(path.split('_')[1])
    
    def get_frames(self, path):
        video = cv2.VideoCapture(self.dirpath + path)
        frames = []
        index = 0
        # how much to add to index each loop
        # AFTER FIRST ROUND OF TRAINING, STEPPER SHOULD BE A CONSTANT AMOUNT
        # FOR REST OF BATCHES
        stepper = int(video.get(cv2.CAP_PROP_FPS) / self.fps)
        print(stepper)
        while index < video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = video.read()
            frames.append(frame)
            index += stepper

        src.release()

        return np.array(frames)[..., [2, 1, 0]]
    
    
    def __call__(self):
        """
        Yields a numpy array of frames, number of white lights
        """
        path, number = self.yield_vids(self)
        yield self.get_frames(path, self.frames, self.fps), number 

testVideoGenerator = VideoGen('batch/train/', 200, 1)

for index, frame in enumerate(testVideoGenerator.get_frames('clip48_2.mp4')):
    imageio.imsave(str(index) + '.png', frame)

#imageio.imsave('./first.png', result[0])
#imageio.imsave('./last.png', result[-1])