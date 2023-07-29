from moviepy.editor import *
import os
import sys
sys.path.append(os.path.abspath(''))
from lib.CONSTANTS import SECS
import math

# constants (EVERYTHING BESIDES COLORS MUST BE EDITED BEFORE EACH BATCH)
WHITE = [255, 255, 255]
RED = [228, 20, 0]

LOC1 = (35, 523)
LOC2 = (35, 554)
LOC3 = (35, 586)
BATCHNUM = 2
LIFTTYPE = "squat"
STARTTIME = 417
ENDTIME = 5749

# initial objects
mainClip = VideoFileClip("USAPL52.mp4").subclip(STARTTIME)
clipseq = 0
index = 1

#ENDTIME = mainClip.duration

def redorwhite(arg) :
    return math.dist(arg, WHITE) < 10 or math.dist(arg, RED) < 50

# index is counted in seconds, counts every third second
while index < ENDTIME :

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
        
        #chopping up clips
        newclip = mainClip.subclip(index - SECS - 2, index - 2)
        newclip = newclip.crop(140, 40, 460, 0)
        newclip.write_videofile(
            LIFTTYPE + "-batch" + str(BATCHNUM) + "-" + str(clipseq) + "_" + str(whiteNum) + ".mp4",
            fps = 24,
            audio = False,
            threads = 10,
            ffmpeg_params=['-f', 'mp4'])
        
        clipseq += 1
        index += 5
    
    index +=1