#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 02:09:41 2019

@author: fanhengxin
"""


import cv2 
import numpy as np




"""
def bi(image):
 
 
    #image：输入图像，可以是Mat类型，
    #       图像必须是8位或浮点型单通道、三通道的图像
    #0：表示在过滤过程中每个像素邻域的直径范围，一般为0
    #后面两个数字：空间高斯函数标准差，灰度值相似性标准差
    dst=cv2.bilateralFilter(image,0,20,10);
    #cv.imshow('bi',dst)
    return dst
 
def shift(image):

 
    #10:空间窗的半径
    #50:色彩窗的半径
    dst=cv2.pyrMeanShiftFiltering(image,15,15);
    #cv.imshow('shift',dst)
    return dst
 
 
 
src=cv2.imread('/Users/fanhengxin/Desktop/yiyi.JPEG') 
 
#图一（原图）
#cv.imshow('def',src)
#图二（色彩窗的半径）
img1=bi(src)
#图三（均值迁移）
img2=shift(src)
cv2.imwrite('/Users/fanhengxin/Desktop/磨皮.jpg',img1)
cv2.imwrite('/Users/fanhengxin/Desktop/油画.jpg',img2)

#if cv2.waitKey(30) & 0xff==ord('q'):
#    cv2.destroyAllWindows()



"""
#import cv2 
#import numpy as np

"""
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dst=cv2.bilateralFilter(frame,0,60,10)
    #dst=cv2.pyrMeanShiftFiltering(dst,10,50)
    # Display the resulting frame
    #height=int(dst.shape[0]/2)
    #width=int(dst.shape[1]/2)
    #frame=cv2.resize(dst,(width,height))
    cv2.imshow('frame',dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""

"""
mg=cv2.imread('/Users/fanhengxin/Desktop/原图.jpg')
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img,[pts],True,(0,255,255))
cv2.imshow('img',img)
if cv2.waitKey(30) & 0xff==ord('q'):
    cv2.destroyAllWindows()
"""



""" Need to increase speed """





img=cv2.imread('/Users/fanhengxin/Desktop/hl.png')
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
    #print('This is the b_sum',b_sum) 
    #print('This is the g_sum',g_sum) 
    #print('This is the r_sum',r_sum)            
    #print('This is the pixel_count',pixel_count)
    final_b=int(round(limit(b_sum/pixel_count)))
    final_g=int(round(limit(g_sum/pixel_count)))
    final_r=int(round(limit(r_sum/pixel_count)))
    #if (final_b != final_g) & (final_b !=final_r):
        #print('Y one pixel found')
        #print([final_b,final_g,final_r])
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

cv2.imwrite('/Users/fanhengxin/Desktop/1_after.png',processed_img)  



    
#cv2.imshow('processed_img',processed_img)
#if cv2.waitKey(1) & 0xff==ord('q'):
#    cv2.destroyAllWindows()
     
      
      





























