# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:44:44 2019

@author: sachin
"""


import cv2
import numpy as np
from numpy import *
import copy

img = cv2.imread('D:/Projects/FYP/Tea Leave Image Evaluation/results/5-croped.jpeg')
h,w,_=img.shape
original_image = copy.copy(img)
cv2.imshow("gray",cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
# Histogram qualization of the image
img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YCrCb2RGB)
# apply HSI color model
hsi_applied_result = cv2.cvtColor(hist_equalization_result, cv2.COLOR_RGB2HLS)
cv2.imshow("h",hsi_applied_result[:,:,0])
cv2.imshow("S",hsi_applied_result[:,:,1])
cv2.imshow("L",hsi_applied_result[:,:,2])
cv2.imwrite('D:/Projects/FYP/image classification/results/L-5.jpg',hsi_applied_result[:,:,2])
Z = hsi_applied_result.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)#cv2.KMEANS_USE_INITIAL_LABELS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
final = res2#cv2.cvtColor(res2, cv2.COLOR_HLS2RGB)

final_gray = cv2.cvtColor(cv2.cvtColor(final, cv2.COLOR_HLS2RGB), cv2.COLOR_RGB2GRAY)
cv2.imshow('final_gray', final_gray)
for i in range(h):
    for j in range(w):
        colorOfPixel=final_gray[i,j]
        #print(colorOfPixel)
        if int(colorOfPixel)>200:
            pass
        else:
            original_image[i,j]=np.array([0,0,0])
cv2.imshow('original_image', original_image)
thresh = 127
im_bw = cv2.threshold(final_gray, thresh, 255, cv2.THRESH_BINARY)[1]
median_blur= cv2.medianBlur(im_bw, 3)
cv2.imshow('median_blur', median_blur)
a=set()
print(final[0,0])
#for x in range(640):
#    for y in range(640):
#        a.add(str(final[x,y]))
#print(a)
lower_g = np.array([30,150,30])
upper_g = np.array([150,200,150])
mask_g = cv2.inRange(final, lower_g, upper_g)
mask_g = cv2.bitwise_not(mask_g)
cv2.imshow('Color input image', img)
cv2.imwrite('D:/Projects/FYP/image classification/results/4.jpg',img)
cv2.imshow('Histogram equalized', hist_equalization_result)
cv2.imwrite('D:/Projects/FYP/image classification/results/4-historgram_equalized.jpg',hist_equalization_result)
cv2.imshow('hsi_applied_result', hsi_applied_result)
cv2.imwrite('D:/Projects/FYP/image classification/results/4-HSI_color_model_applied-1.jpg',hsi_applied_result)
cv2.imshow('final',final)
cv2.imwrite('D:/Projects/FYP/image classification/results/4-K_means_clusterisation_applied.jpg',final)
cv2.imshow('final_gray',final_gray)
cv2.imwrite('D:/Projects/FYP/image classification/results/4-grayed_clusterd_image.jpg',final_gray)
cv2.imshow('im_bw',im_bw)
cv2.imwrite('D:/Projects/FYP/image classification/results/4-Threashhold_applied.jpg',im_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result.jpg',hist_equalization_result)
print(1)
