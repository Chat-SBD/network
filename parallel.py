from tensorflow import keras

from lib.Server import Server

server = Server(
    'squat/model/modelfile',
    keras.optimizers.Adam(),
    keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    'lifts/squat/batch/progress.txt'
)

server.train(secs = 25, fps = 24, seed = 42)