import numpy as np
import cv2
img=cv2.imread(''). #Change it to your input img address
radius=1
height=img.shape[0]
width=img.shape[1]
processed_img=np.copy(img)
for row in range(radius,height-radius):
    for col in range(radius,width-radius):
        b_neighbour=img[row-radius:row+radius+1,col-radius:col+radius+1,0]
        g_neighbour=img[row-radius:row+radius+1,col-radius:col+radius+1,1]
        r_neighbour=img[row-radius:row+radius+1,col-radius:col+radius+1,2]
        value_list=[int(min(255,np.median(b_neighbour))),int(min(255,np.median(g_neighbour))),int(min(255,np.median(r_neighbour)))]
        #print(value_list)
        for i in range(3):
            processed_img[row,col,i]=value_list[i]
processed_img=np.uint8(processed_img)
cv2.imwrite('',processed_img) #Change it to your output img address

cv2.imshow('denoised',processed_img)
cv2.waitKey(1)

