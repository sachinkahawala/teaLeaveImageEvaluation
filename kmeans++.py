# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:44:44 2019

@author: sachin
"""

import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Read
img = io.imread("D:/Projects/FYP/image classification/results/finalResultsWithOutBuds.jpg")

# Split
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]
#print(type(green))
# blue=[[0]*1152 for _ in range(576)]
# blue = np.asarray(blue)
# img[:,:,2]=blue
#print(type(blue),len(blue),len(blue[0]))
#plt.imshow(green, cmap='Greens')
plt.imshow(img)
# Plot
fig, axs = plt.subplots(2,2)

cax_00 = axs[0,0].imshow(img)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

cax_01 = axs[0,1].imshow(red, cmap='Reds')
fig.colorbar(cax_01, ax=axs[0,1])
axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

cax_10 = axs[1,0].imshow(green, cmap='Greens')
fig.colorbar(cax_10, ax=axs[1,0])
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

cax_11 = axs[1,1].imshow(blue, cmap='Blues')
fig.colorbar(cax_11, ax=axs[1,1])
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()

# Plot histograms
fig, axs = plt.subplots(3, sharex=True, sharey=True)

axs[0].hist(red.ravel(), bins=10)
axs[0].set_title('Red')
axs[1].hist(green.ravel(), bins=10)
axs[1].set_title('Green')
axs[2].hist(blue.ravel(), bins=10)
axs[2].set_title('Blue')

plt.show()
