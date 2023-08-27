from moviepy.editor import VideoFileClip
from lib.CONSTANTS import SECS, FPS, SIZE
import math

# constants (EVERYTHING BESIDES COLORS MUST BE EDITED BEFORE EACH BATCH)
WHITE = [255, 255, 255]
RED = [228, 20, 0]

# (y, x)
LOC1 = (283, 112)
LOC2 = (283, 131)
LOC3 = (283, 150)
BATCHNUM = 12
LIFTTYPE = "squat"
STARTTIME = 1140
OFFSET = 2

# upper left corner of crop
X1 = 175
Y1 = 5

# initial objects
mainClip = VideoFileClip("C:/Users/samed/Downloads/videoplayback (10).mp4").subclip(STARTTIME)
clipseq = 0
index = 1

ENDTIME = 5700
#ENDTIME = mainClip.duration

def redorwhite(arg) :
    return math.dist(arg, WHITE) < 10 or math.dist(arg, RED) < 50

# index is counted in seconds, counts every third second
while index < ENDTIME - STARTTIME:
    light1 = mainClip.get_frame(index)[LOC1]
    light2 = mainClip.get_frame(index)[LOC2]
    light3 = mainClip.get_frame(index)[LOC3]

    allLights = [light1, light2, light3]

    whiteNum = 0
    
    if all(map(redorwhite, allLights)) :
        # incrementing counter of white lights for each white light found
        for val in allLights :
            if (math.dist(val, WHITE) < 10) :
                whiteNum += 1
        
        # chopping up clips
        newclip = mainClip.subclip(index - SECS - OFFSET, index - OFFSET)
        newclip = newclip.crop(x1 = X1, y1 = Y1, x2 = X1 + SIZE, y2 = Y1 + SIZE)
        newclip.write_videofile(
            LIFTTYPE + "-vid" + str(BATCHNUM) + "-" + str(clipseq) + "_" + str(whiteNum) + ".mp4",
            fps = FPS,
            audio = False,
            threads = 10,
            ffmpeg_params=['-f', 'mp4'])
        
        clipseq += 1
        index += 5
    
    index +=1