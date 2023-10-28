from glob import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns

# add below to each file for imports :|
import os
import sys

sys.path.append(os.path.abspath(''))
# add above to each file for imports :|

# constants
from lib.CONSTANTS import SECS, FPS, TEST_SIZE, VAL_SIZE

def get_vids(path):
    """
    Gets an array of tuples of full relative video path and number of white lights
    [('batch/train/squat-batch1-18_2', 2), (...]

    Args:
        path: str. 'batch/train/' or 'batch/test/', whichever is being used.
    """
    videos = glob(path + '*.mp4')

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

def train_test_val(x, y, test_size = TEST_SIZE, val_size = VAL_SIZE):
    """
    Return x_train, y_train, x_test, y_test, x_val, y_val.

    Args:
        x: array. Data to be classified.
        y: array. Corresponding classification labels.
        test_size: float. The percent size of all data to put into test (0.1, 0.2, etc).
        val_size: float. The percent size of the train batch to put into val.
    """
    test_index = int(len(x) * test_size) - 1
    x_test = x[0 : test_index]
    y_test = y[0 : test_index]
    x_train = x[test_index :]
    y_train = y[test_index :]

    val_index = int(len(x_train) * val_size) - 1
    x_val = x_train[0 : val_index]
    y_val = y_train[0 : val_index]
    x_train = x_train[val_index :]
    y_train = y_train[val_index :]

    return x_train, y_train, x_test, y_test, x_val, y_val

def expand(x, y):
    """
    Duplicate data to make each category the same size.
    Keep in mind that the data will be organized throughout the arrays into each category,
    and will need to be re-shuffled.

    Args:
        x: array. Data to be classified.
        y: array. Corresponding classification labels.
    """

    unique, counts = np.unique(y, return_counts = True)
    max_unique = np.max(counts)

    new_x = np.ndarray(shape = (0))
    new_y = np.ndarray(shape = (0))

    # for each unique category...
    for cat in unique:
        expanded = []

        videos = []
        index = 0
        # for each video in x...
        while index < len(x):
            # if that video is in the category...
            if y[index] == cat:
                videos.append(x[index])
            index += 1
        
        # while the expanded category array is not long enough...
        while len(expanded) < max_unique:
            # for each video in that category...
            for video in videos:
                # if the expanded array is STILL not long enough yet...
                if len(expanded) < max_unique:
                    expanded.append(video)
                else:
                    break
        
        # add the expanded array to the full expanded x, and to y
        new_x = np.concatenate((new_x, np.array(expanded)), axis = 0)
        new_y = np.concatenate((new_y, np.array([cat] * len(expanded))), axis = 0)
    
    return new_x, new_y

def compress(x, y):
    """
    Delete data to make each category the same size.
    Keep in mind that the data will be organized throughout the arrays into each category,
    and will need to be re-shuffled.

    Args:
        x: array. Data to be classified.
        y: array. Corresponding classification labels.
    """

    unique, counts = np.unique(y, return_counts = True)
    min_unique = np.min(counts)

    new_x = np.ndarray(shape = (0))
    new_y = np.ndarray(shape = (0))

    # for each unique category...
    for cat in unique:
        expanded = []

        videos = []
        index = 0
        # for each video in x...
        while index < len(x):
            # if that video is in the category...
            if y[index] == cat:
                videos.append(x[index])
            index += 1
        
        # while the expanded category array is not long enough...
        while len(expanded) < min_unique:
            # for each video in that category...
            for video in videos:
                # if the expanded array is STILL not long enough yet...
                if len(expanded) < min_unique:
                    expanded.append(video)
                else:
                    break
        
        # add the expanded array to the full expanded x, and to y
        new_x = np.concatenate((new_x, np.array(expanded)), axis = 0)
        new_y = np.concatenate((new_y, np.array([cat] * len(expanded))), axis = 0)
    
    return new_x, new_y

def variate(vid):
    """
    Randomly increases or decreases every integer in a (FRAMES, SIZE, SIZE, 1) array,
    keeping them inclusive between 0 and 255.
    This adds variation to expanded datasets to prevent overfitting

    Args:
        vid: array. Shape = (FRAMES, SIZE, SIZE, 1)
    """

    randoms = np.random.randint(-10, 11, size = vid.shape)
    return np.clip(vid + randoms, 0, 255)

def acc(actual_values, predicted_values):
    correct_predictions = 0

    for actual, predicted in zip(actual_values, predicted_values):
        if int(actual) == int(predicted):
            correct_predictions += 1

    accuracy = correct_predictions / len(actual_values)
    return accuracy

def plot_cm(conf_matrix, class_names, filename):
    plt.figure(figsize = (8, 6))
    sns.set(font_scale = 1.2)
    sns.heatmap(
        conf_matrix,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        xticklabels = class_names,
        yticklabels = class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()