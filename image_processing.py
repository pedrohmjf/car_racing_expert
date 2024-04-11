import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
import math

def color_filter(observation):
  cropped = observation[63:65, 24:73]

  hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
  mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

  # Slice the green
  imask_green = mask_green>0
  green = np.zeros_like(cropped, np.uint8)
  green[imask_green] = cropped[imask_green]
  return green

def canny(green):
  canny = cv2.Canny(green, 50, 150)
  return canny

########################
#### PID Controller ####
########################

# Find the middle of the lane
def find_middle(canny, previous_error):
    nz = cv2.findNonZero(canny)
    mid  = 24
    if nz[:,0,0].max() == nz[:,0,0].min():
        if nz[:,0,0].max() <30 and nz[:,0,0].max()>20:
            return previous_error
        if nz[:,0,0].max() >= mid:
            return(-15)
        else:
            return(+15)
    else:
        return(((nz[:,0,0].max() + nz[:,0,0].min())/2)-mid)

# PID equation
def pid(error, previous_error):
    Kp = 0.02
    Ki = 0.07
    Kd = 0.2

    steering = Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)

    return steering