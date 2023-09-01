from moviepy.editor import VideoFileClip
from lib.CONSTANTS import SECS, FPS, SIZE
import math

# constants (EVERYTHING BESIDES COLORS MUST BE EDITED BEFORE EACH BATCH)
WHITE = [255, 255, 255]
RED = [228, 20, 0]

# (y, x)
LOC1 = (327, 118)
LOC2 = (327, 138)
LOC3 = (327, 157)
BATCHNUM = 13
LIFTTYPE = "deadlift"
STARTTIME = 11940
OFFSET = 4

# upper left corner of crop
X1 = 197
Y1 = 6

# initial objects
mainClip = VideoFileClip("C:/Users/samed/Downloads/videoplayback (1).mp4").subclip(STARTTIME)
clipseq = 0
index = 1

ENDTIME = 15780
#ENDTIME = STARTTIME + mainClip.duration

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
        index += 10
    
    index +=1