import cv2
from glob import glob
import numpy as np
import random
import tensorflow as tf
import einops
import keras
from keras import layers

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

# representing a layer for the 2+1D network
class Conv2Plus1D(keras.layers.Layer) :
    def __init__(self, filters, kernel_size, padding):
        """
        A sequence of convolutional layers that first apply the convolution operation over the
        spatial dimensions, and then the temporal dimension. 
        """
        super().__init__()
        #sequences the spatial and temporal decompositions into a model
        self.seq = keras.Sequential([
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

    def call(self, x) :
        return self.seq(x)


class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)


class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)  
  

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos



# building the layers of our residual network
input_shape = (None, 28, 640, 640, 3)
input = layers.Input(shape = (input_shape[1:]))
x = input

x = Conv2Plus1D(filters = 16, kernel_size = (3, 7, 7), padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(640 // 2, 640 // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(640 // 4, 640 // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(640 // 8, 640 // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(640 // 16, 640 // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(4)(x)

model = keras.Model(input, x)




output_sig = (
    tf.TensorSpec(shape = (28, 640, 640, 3), dtype = tf.int16),
    tf.TensorSpec(shape = (), dtype = tf.int8)
)

train_dsv1 = tf.data.Dataset.from_generator(
    VideoGen('batch/train/', 28, 1),
    output_signature = output_sig
)

test_dsv1 = tf.data.Dataset.from_generator(
    VideoGen('batch/test/', 28, 1),
    output_signature = output_sig
)



#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

frames, labels = next(iter(train_ds))
print(frames.shape)

model.build(frames)
# Visualize the model
keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True)


model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
              metrics = ['accuracy', 'FalseNegatives', 'FalsePositives'])



history = model.fit(x = train_ds,
                    epochs = 50, 
                    validation_data = test_ds)

