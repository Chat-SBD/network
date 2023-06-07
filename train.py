import cv2
import imageio
import numpy as np

VIDDIR = 'UCF101/'
VIDPATH = 'clip48_2.mp4'

src = cv2.VideoCapture(VIDPATH)
result = [frame for ret, frame in iter(src.read, (False, None))]
src.release()

result = np.array(result)[..., [2, 1, 0]]

imageio.imsave('./first.png', result[0])
imageio.imsave('./last.png', result[-1])