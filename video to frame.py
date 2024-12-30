# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:53:36 2023

@author: cse
"""

import torch
import cv2
import os
import time
import numpy as np

# Read the video from specified path
cam = cv2.VideoCapture("3urban.mp4")



# Set the yolo5 model to detect person only
model.classes = [0]  # detect only persons


# Get the sample video here
video = cv2.VideoCapture('3urban.mp4')

# Get the frame rate
fps = video.get(cv2.CAP_PROP_FPS)


# Create a folder to keep the frames of extracted video  
try:
      
    # creating a folder named data
    if not os.path.exists('FRAMES'):
        os.makedirs('FRAMES')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame rate to be changed - frames taken at each second
currentframe = 0

# Sequence for frames to be written - relevant frames
newframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
    
    # Clear Console
    print("\033[H\033[J") # Just to clear all console outputs
  
    if ret:
        # if video is still left continue creating images
        # save frame
        name = 'FRAMES/frame' + str(newframe) + '.jpg'
        print ('Creating...' + name)
        
        newframe = newframe + 1
  
        # writing the extracted images
        cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
        #currentframe += fps # i.e. at 30 fps, this advances one second
        #cam.set(1, currentframe)
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()


