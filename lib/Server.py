from glob import glob
from mpi4py import MPI
from tensorflow import keras

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

from lib.data import get_vids, get_frames
from lib.network import gradient, dataset
from lib.CONSTANTS import SEED, SECS, FPS, STATUS, GRAD

class Server:
    def __init__(self, modelfolder, liftfolder, optimizer, lossf):
        """
        A class to manage cluster nodes, serve data to them, and average/apply their gradients.
        Every node instantiates this class for itself, but only the master node can perform certain operations.
        It works, just trust

        Args:
            modelfolder: str. The location of the folder of the model file that you are trying to train. 'lifts/squat/models/conv21d/'
            liftfolder: str. The location of the lift folder. 'lifts/squat/'
            optimizer: tensorflow.keras.Optimizer.
            lossf: tensorflow.keras.Loss.
        """
        self.progpath = modelfolder + 'progress'
        self.modelpath = modelfolder + 'model'
        self.liftfolder = liftfolder

        self.world = MPI.COMM_WORLD
        self.rank = self.world.Get_rank()
        self.nprocs = self.world.Get_size()

        self.model = keras.models.load_model(self.modelpath)
        self.optimizer = optimizer
        self.lossf = lossf

        with open(self.progpath, 'r') as progfile:
            self.start = int(progfile.readline().strip())
        self.log('Made Server')

        error = 10 / 0
    
    def log(self, info):
        if self.rank == 0:
            with open('train.log', 'a') as file:
                file.write(info)
                file.write('\n')
    
    def progress(self, index):
        """
        Write a number to a file. Only the master process (rank 0) can execute, to limit IO.

        Args:
            index: any. The int, usually, to overwrite to the file.
        """
        if self.rank == 0:
            with open(self.progpath, 'w') as progfile:
                progfile.write(str(index))
    
    def gradient(self, secs = SECS, fps = FPS, seed = SEED):
        """
        Distribute videos to all processes according to rank.
        Average their calculated gradient descents and yield them in batches, currently of 10.

        Args:
            secs: int. The number of seconds in the video to read in.
            fps: int. Frames per second to read in.
            seed: int = SEED. Random seed to shuffle videos with.
        """
        videos = get_vids(self.liftfolder + 'batch/train/', seed)
        vidindex = self.start
        self.log('Got videos')

        # for each set of videos...
        while vidindex < len(videos):
            # my personal index as a process
            myindex = vidindex + self.rank

            # if i have a video this loop...
            if myindex < len(videos):
                # let master process know
                self.world.send(True, dest = 0, tag = STATUS)

                path, lights = videos[myindex]
                self.log(path)
                myvid = get_frames(path, secs, fps)

                self.log('Got frames')
                # send my gradient to master process
                self.world.send(
                    gradient(
                        self.model,
                        self.lossf,
                        dataset(myvid, lights)
                    ),
                    dest = 0,
                    tag = GRAD
                )
                self.log('Sent')
            
            else:
                self.world.send(False, dest = 0, tag = STATUS)

            # if i am master process...
            if self.rank == 0:
                ngrads = 0
                sumgrad = 0

                # for each process...
                for rank in range(self.nprocs):
                    # if that rank processed a gradient...
                    if self.world.recv(source = rank, tag = STATUS):
                        ngrads += 1
                        sumgrad += self.world.recv(source = rank, tag = GRAD)
                
                self.log(f'Sum of gradients: {sumgrad}')
                self.log(f'Number of gradients: {ngrads}')
                grad = sumgrad / ngrads
                yield grad
                vidindex += ngrads
            
            # sync index across all nodes after master updates it
            vidindex = self.world.bcast(vidindex, root = 0)
            print('-----------------------------------------BATCH---------------------------------')
            self.progress(vidindex)
    
    def train(self, secs = SECS, fps = FPS, seed = SEED):
        """
        Train the model. For each gradient from self.gradient, apply it to the model and save it.
        """
        for gradient in self.gradient(secs, fps, seed):
            if self.rank == 0:
                self.optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))
                self.model.save(self.modelpath)