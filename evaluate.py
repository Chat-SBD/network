"""
Evaluate the model.

Args:
    1: str. The path to the saved model folder. 'lifts/squat/models/conv21d/'
"""
import os
from sys import argv
from tensorflow import keras
import tensorflow as tf

from lib.data import acc, plot_cm
from lib.network import FrameGenerator
from lib.CONSTANTS import FRAMES, SIZE

SAVEPATH = argv[1] + 'model/'
DSPATH = '/'.join(argv[1].split('/')[: 2]) + '/dataset/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Loading model...')
model = keras.models.load_model(SAVEPATH)

print('Building dataset...')
outsig = (
    tf.TensorSpec(shape = (FRAMES, SIZE, SIZE, 1), dtype = tf.int16),
    tf.TensorSpec(shape = (), dtype = tf.int16)
)
fg = FrameGenerator(DSPATH, 'test')
ds_test = tf.data.Dataset.from_generator(fg, output_signature = outsig)
ds_test = ds_test.batch(100)

print('Evaluating...')
actual = tf.stack(fg.y, axis = 0)
predicted = tf.argmax(tf.concat(model.predict(ds_test), axis = 0), axis = 1)
accuracy = acc(actual, predicted)

with open(argv[1] + 'acc.txt', 'w') as file:
    file.write(str(accuracy))

print(f'Accuracy: {accuracy}')

print('Making confusion matrices...')
cm = tf.math.confusion_matrix(actual, predicted)
plot_cm(cm, [0, 1, 2, 3], argv[1] + 'cm_all.png')

cm_binary = cm.numpy().reshape(2, 2, 2, 2).sum(axis = (2, 3))
plot_cm(cm_binary, ['no lift', 'lift'], argv[1] + 'cm_binary.png')