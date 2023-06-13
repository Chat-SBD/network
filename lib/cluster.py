import cv2
from glob import glob
import numpy as np
import random
from mpi4py import MPI

# constants
TRAIN = 'batch/train/'
TEST = 'batch/test'
PROG = 'batch/progress.txt'
SEED = 42
SECS = 25
FPS = 24

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

    return zip(videos, int(videos.split('_')[1].split('.')[0]))

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
    def __init__(self, progpath = PROG):
        """
        A class to manage cluster nodes, serve videos to them, and average/apply their gradients.

        Args:
            path: str. The relative path to a file to write training progress to. 'batch/progress.txt'
        """
        self.progpath = progpath
        self.world = MPI.COMM_WORLD
        self.rank = self.world.Get_rank()
        self.nprocs = self.world.Get_size()
        with open(self.progpath, 'r') as progfile:
            self.start = int(progfile.readline().strip())
    
    def progress(self, index):
        """
        Write a number to a file. Only the master process (rank 0) can execute, to limit IO.

        Args:
            index: any. The int, usually, to overwrite to the file.
        """
        if self.rank == 0:
            with open(self.progpath, 'w') as progfile:
                progfile.write(str(index))
    
    def train(self, secs = SECS, fps = FPS, seed = SEED):
        """
        Distribute videos to all processes according to rank.
        Average their calculated gradient descents, apply them,
        and update progress index.

        Args:
            secs: int. The number of seconds in the video to read in.
            fps: int. Frames per seconds to read in.
            seed: int = SEED. Random seed to shuffle videos with.
        """
        videos = get_vids(TRAIN, seed)
        vidindex = self.start
        # for each set of videos...
        while vidindex < len(videos):
            # my personal index as a process
            myindex = vidindex + self.rank

            # if i have a video this loop...
            if myindex < len(videos):
                path, lights = videos[myindex]
                myvid = get_frames(path, secs, fps)
                mygradient = 'do something here'
                # send my gradient to master process
                self.world.send(mygradient, dest = 0)

            # if i am master process...
            if self.rank == 0:
                ngradients = 0
                sumgradient = 0
                # for each process...
                for rank in range(self.nprocs):
                    # if that rank processed a gradient...
                    if self.world.iprobe(source = rank):
                        ngradients += 1
                        sumgradient += self.world.recv(source = rank)
                gradient = sumgradient / ngradients
                # apply gradient to model
                apply = 'do something here'
            
            vidindex += self.nprocs
            self.progress(vidindex)