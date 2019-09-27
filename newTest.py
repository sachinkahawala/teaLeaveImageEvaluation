import cv2
import numpy as np

img = cv2.imread('D:/Projects/FYP/finalResultsWithBuds-croped.jpg', 0)
_, blackAndWhite = cv2.threshold(img, 175, 200, cv2.THRESH_BINARY_INV)
cv2.imshow('blackAndWhite', blackAndWhite)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 4, cv2.CV_32S)
sizes = stats[1:, -1] #get CC_STAT_AREA component
img2 = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if sizes[i] >= 50:   #filter small dotted regions
        img2[labels == i + 1] = 255

res = cv2.bitwise_not(img2)

cv2.imshow('res.png', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
