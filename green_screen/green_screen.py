#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Mon Aug 26 12:39:30 2019



import numpy as np
import cv2
import os
import sys
import random
from matplotlib import pyplot as plt

BLUE = 120

if not os.path.isdir(os.path.join(os.getcwd(), 'background')):
    os.mkdir("background")
else:
    print('background already exists')

if not os.path.isdir(os.path.join(os.getcwd(), 'composite')):
    os.mkdir("composite")
else:
    print('composite already exists')

cap = cv2.VideoCapture('') #This is the background video input 
if not cap.isOpened():
    print('Quadrangle.mov not opened')
    sys.exit(1)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
bgctr = 1 # The total number of background frames
count = 0
while(1):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite('background/frame%d.tif' % count, frame)
    count += 1
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

bgctr = count
count = 0
cap = cv2.VideoCapture('') #This is the foreground video
if not cap.isOpened():
    print('monkey.mov not opened')
    sys.exit(1)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while(1):
    ret, monkeyframe = cap.read()
    if not ret:
        break
    bg = cv2.imread('background/frame%d.tif' % (count%bgctr))
    if bg is None:
        print('ooops! no bg found BG/frame%d.tif' % (count%bgctr))
        break
    # overwrite the background
    for x in range(monkeyframe.shape[0]):
        for y in range(monkeyframe.shape[1]):
            if monkeyframe[x][y][0] < BLUE:
                bg_x = int((x/monkeyframe.shape[0])*bg.shape[0])
                bg_y = int((y/monkeyframe.shape[1])*bg.shape[1])
                for ind in range(3):
                    bg[bg_x][bg_y][ind] = monkeyframe[x][y][ind]
    cv2.imwrite('composite/composite%d.tif' % count, bg)
    cv2.putText(img=bg, text='Compositing: %d%%' % int(100*count/length), org=(int(0), int(bg.shape[1] / 2)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7,
                color=(0, 255, 0))
    cv2.imshow('Monkey in Quadrangle', bg)

    count += 1
    if cv2.waitKey(30) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

count = 0
out = cv2.VideoWriter('monkey_dance.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (int(frame_width), int(frame_height)))
while(1):
    img = cv2.imread('composite/composite%d.tif' % count)
    if img is None:
        break
    out.write(img)
    count += 1
out.release()
cv2.destroyAllWindows()
