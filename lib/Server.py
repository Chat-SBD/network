from mpi4py import MPI
from tensorflow import keras

from lib.data import SEED, SECS, FPS, get_vids, get_frames
from lib.network import gradient

# constants
TRAIN = 'lifts/squat/batch/train/'
TEST = 'lifts/squat/batch/test'
PROG = 'lifts/squat/batch/progress.txt'
STATUS = 111
GRAD = 222

class Server:
    def __init__(self, modelpath, optimizer, lossf, progpath = PROG):
        """
        A class to manage cluster nodes, serve data to them, and average/apply their gradients.
        Every node instantiates this class for itself, but only the master node can perform certain operations.
        It works, just trust

        Args:
            modelpath: str. The location of the file of the model that you are looking to train.
            optimizer: tensorflow.keras.Optimizer.
            lossf: tensorflow.keras.Loss.
            progpath: str. The relative path to a file to write training progress to. 'batch/progress.txt'
        """
        self.progpath = progpath
        self.modelpath = modelpath

        self.world = MPI.COMM_WORLD
        self.rank = self.world.Get_rank()
        self.nprocs = self.world.Get_size()

        self.model = keras.models.load_model(modelpath)
        self.optimizer = optimizer
        self.lossf = lossf

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
    
    def gradient(self, secs = SECS, fps = FPS, seed = SEED):
        """
        Distribute videos to all processes according to rank.
        Average their calculated gradient descents and yield them in batches, currently of 10.

        Args:
            secs: int. The number of seconds in the video to read in.
            fps: int. Frames per second to read in.
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
                # let master process know
                self.world.send(True, dest = 0, tag = STATUS)

                path, lights = videos[myindex]
                myvid = get_frames(path, secs, fps)

                # send my gradient to master process
                self.world.send(gradient(self.model, self.lossf, (myvid, lights)), dest = 0, tag = GRAD)
            
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
                
                gradient = sumgrad / ngrads
                yield gradient
                vidindex += ngrads
            
            # sync index across all nodes after master updates it
            vidindex = self.world.bcast(vidindex, root = 0)

            self.progress(vidindex)
    
    def train(self, secs = SECS, fps = FPS, seed = SEED):
        """
        Train the model. For each gradient from self.gradient, apply it to the model and save it.
        """
        if self.rank == 0:
            for gradient in self.gradient(secs = SECS, fps = FPS, seed = SEED):
                self.optimizer.apply_gradients(zip(gradient, self.model.trainable_weights))
                self.model.save(self.modelpath)