#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 02:09:41 2019
@author: fanhengxin
"""

# you need to install numpy & cv2 in your computer
import cv2 
import numpy as np

img=cv2.imread('xxx') #change it to your picture address, preferable absolute address
height=img.shape[0]
width=img.shape[1]
kernel_size=7
kernel_intensity=20
def limit(pixel_value):
    return min(pixel_value,255)

def get_most_shown_color(img,row,col,radius):
    neighbour=img[row-radius:row+radius+1,col-radius:col+radius+1,:]
    pixel_intensity=np.zeros((neighbour.shape[0],neighbour.shape[1]),dtype='int64')
    #print('This is the original pixel_intensity matrix',pixel_intensity)
    for h in range(neighbour.shape[0]):
        for w in range(neighbour.shape[1]):
            intensity=round((int(neighbour[h,w,0])+int(neighbour[h,w,1])+int(neighbour[h,w,2]))/3*20/255)
            pixel_intensity[h,w]=intensity
            #print(pixel_intensity)
    #print('This is pixel_intensity',pixel_intensity)
    flat=pixel_intensity.flatten()
    flat= flat.astype('int64') 
    count=np.bincount(flat)
    most_shown_value=np.argmax(count)
    pixel_count=np.amax(count)
    b_sum=0;g_sum=0;r_sum=0
    for h_ in range(pixel_intensity.shape[0]):
        for w_ in range(pixel_intensity.shape[1]):
            if pixel_intensity[h_,w_]==most_shown_value:
                b_sum+=neighbour[h_,w_,0]
                g_sum+=neighbour[h_,w_,1]
                r_sum+=neighbour[h_,w_,2]
    final_b=int(round(limit(b_sum/pixel_count)))
    final_g=int(round(limit(g_sum/pixel_count)))
    final_r=int(round(limit(r_sum/pixel_count)))
    return (final_b,final_g,final_r)

def padding_helper(img,pad_size):
    pad_dim=((pad_size,pad_size),(pad_size,pad_size),(0,0))
    padded_img=np.pad(img,pad_dim,mode='constant',constant_values=0)
    return np.uint8(padded_img)


#padded_img=padding_helper(img,int(kernel_size/2))
processed_img=np.zeros(img.shape,dtype='int64')
radius=int(kernel_size/2)
for row in range(radius,height-radius):
    for col in range(radius,width-radius):
        pixel_color_list=get_most_shown_color(img,row,col,radius)
        #print('One pixel done')
        #print('the pixel_color_list is',pixel_color_list)
        for i in range(3):
            processed_img[row,col,i]=pixel_color_list[i]
        #print("Origin,col_%d " % col,img[row,col,:] )
        #print('Processed,col_%d' % col,processed_img[row,col,:])
        #print(processed_img[row,col,:])
processed_img=np.uint8(processed_img)

cv2.imwrite('xxx',processed_img)  #Change xxx to the place that you would like to store it.



#This step is to display image    
cv2.imshow('processed_img',processed_img)
if cv2.waitKey(1) & 0xff==ord('q'):
    cv2.destroyAllWindows()
