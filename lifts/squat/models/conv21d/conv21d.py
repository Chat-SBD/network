import tensorflow as tf
from tensorflow import keras
import einops

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

from lib.data import get_frames
from lib.network import dataset
from lib.CONSTANTS import FRAMES

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
            keras.layers.Conv3D(
                filters = filters,
                kernel_size = (1, kernel_size[1], kernel_size[2]),
                padding = padding
            ),
            # Temporal decomposition
            keras.layers.Conv3D(
                filters = filters, 
                kernel_size = (kernel_size[0], 1, 1),
                padding = padding)
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
            Conv2Plus1D(
                filters = filters,
                kernel_size = kernel_size,
                padding = 'same'
            ),
            keras.layers.LayerNormalization(),
            keras.layers.ReLU(),
            Conv2Plus1D(
                filters = filters, 
                kernel_size = kernel_size,
                padding = 'same'
            ),
            keras.layers.LayerNormalization()
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
            keras.layers.Dense(units),
            keras.layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)  

def add_residual_block(input, filters, kernel_size):
    """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters, kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return keras.layers.add([res, out])

class ResizeVideo(keras.layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = keras.layers.Resizing(self.height, self.width)

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
input_shape = (None, FRAMES, 640, 640, 1)
input = keras.layers.Input(shape = (input_shape[1:]))
x = input

x = Conv2Plus1D(filters = 4, kernel_size = (3, 7, 7), padding = 'same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
x = ResizeVideo(640 // 2, 640 // 2)(x)

# Block 1
x = add_residual_block(x, 8, (3, 3, 3))
x = ResizeVideo(640 // 4, 640 // 4)(x)

# Block 2
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(640 // 8, 640 // 8)(x)

# Block 3
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(640 // 16, 640 // 16)(x)

# Block 4
x = add_residual_block(x, 64, (3, 3, 3))

x = keras.layers.GlobalAveragePooling3D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(4)(x)

model = keras.Model(input, x)
print('---------------------------------------------------Created model---------------------------------------------------')

model.build(dataset(get_frames('lifts/squat/models/squat-sample.mp4'), 0))
print('----------------------------------------------------Built model----------------------------------------------------')

model.save('lifts/squat/models/conv21d/model')
keras.utils.plot_model(model, to_file = 'lifts/squat/models/conv21d/conv21d.png', show_shapes = True)
print('----------------------------------------------------Saved model----------------------------------------------------')