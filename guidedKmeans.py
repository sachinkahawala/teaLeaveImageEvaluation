
import cv2
import numpy as np
from math import sqrt as squareRoot
import time
from math import isnan
import copy
#print 1
def distance(pointA, pointB):
    try :
        pointA[1]
        return (
            ((pointA[0] - pointB[0]) ** 2) +
            ((pointA[1] - pointB[1]) ** 2) +
            ((pointA[2] - pointB[2]) ** 2)
        )
    except Exception as e:
        #print(1)
        return (pointA-pointB)**2
def imshow_components(image,original_image):
    h,w,_=original_image.shape
    img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    img = img.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[:, -1]
    img2 = np.zeros(output.shape)

    for i in range(1, nb_components):
        print(sizes[i])
        if sizes[i] > 50:
            img2[output == i] = 255
    cv2.imshow("Biggest component", img2)
    cv2.imshow("image", image)
    for i in range(h):
        for j in range(w):
            if img2[i,j]:
                pass
            else:
                original_image[i,j]=np.array([0,0,0])
    cv2.imshow("original_image", original_image)
    #cv2.waitKey()


def guidedKMEANS(image,k,values,iterations):
    original_image = copy.copy(image)
    h,w,_=image.shape
    print(h,w)
    img_to_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    hist_equalization_result = cv2.cvtColor(img_to_yuv, cv2.COLOR_YCrCb2RGB)
    cv2.imshow('hist_equalization_result',hist_equalization_result)
    cv2.imwrite('D:/Projects/FYP/Tea Leave Image Evaluation/results/hist_equalization_result.jpg',hist_equalization_result)
    # apply HSI color model
    hsi_applied_result = cv2.cvtColor(hist_equalization_result, cv2.COLOR_RGB2HLS)
    cv2.imwrite('D:/Projects/FYP/image classification/results/Hue.jpg',hsi_applied_result[:,:,0])
    cv2.imwrite('D:/Projects/FYP/image classification/results/Light.jpg',hsi_applied_result[:,:,1])
    cv2.imwrite('D:/Projects/FYP/image classification/results/Saturation.jpg',hsi_applied_result[:,:,2])
    cv2.imshow('hsi_applied_result',hsi_applied_result)
    image=hsi_applied_result[:,:,2]
    #print(image)
    print(h,w)
    clusterMat=[[-1]*w for p in range(h)]
    for iteration in range(iterations):
        start=time.time()
        VVV=0
        for i in range(h):
            for j in range(w):
                cBest=1231313123
                for cluster in range(k):
                    start2=time.time()
                    imV=image[i,j]
                    value=values[cluster]
                    #imV = np.array(imV)
                    #value = np.array(value)
                    #dist= np.sqrt(np.sum((imV-value)**2))
                    #dist = np.linalg.norm(imV-value)
                    dist=distance(imV,value)
                    VVV+=time.time()-start2
                    if dist<cBest:
                        cBest=dist
                        clusterMat[i][j]=cluster

        print(str(VVV)+"   SS")
        dic={}
        for i in range(k):
            dic[i]=[]
        for i in range(h):
            for j in range(w):
                dic[clusterMat[i][j]].append((image[i][j]))
        newValues=[]
        for cluster in range(k):
            newCluster=np.mean(dic[cluster],axis=0)

            if len(dic[cluster])==0:
                newValues.append(values[cluster])
            else:
                newValues.append(newCluster)
        values=newValues[:]
        print("iteration "+str(iteration))
        print(newValues)
        for cluster in range(k):

            print(str(cluster)+" number "+str(len(dic[cluster])))
        print()
        print(time.time()-start)
    colors=[]
    for t in range(k):
        colors.append(t*255/k)
    #clone_img = copy.copy(image)

    for i in range(h):
        for j in range(w):
            colorOfPixel=colors[clusterMat[i][j]]
            if clusterMat[i][j]==k-1:
                pass
                #original_image[i,j]=np.array([colorOfPixel,colorOfPixel,colorOfPixel])
            else:
                original_image[i,j]=np.array([0,0,0])
            #print(colorOfPixel)
            if type(image[i,j])==np.uint8:
                image[i,j]=colorOfPixel
            else:
                image[i,j]=np.array([colorOfPixel,colorOfPixel,colorOfPixel])
    cv2.imshow('image12',image)
    cv2.imshow('original_image1',original_image)
    for i in range(h):
        for j in range(w):
            #remove reflected pixels
            # if original_image[i,j][2]<150:
            #     original_image[i,j]=np.array([0,0,0])
            #     #remove noise
            if original_image[i,j][1]<150:
                original_image[i,j]=np.array([0,0,0])
                image[i,j]=np.array([0])
    cv2.imshow('image1',image)
    cv2.imshow('original_image2',original_image)
    #imshow_components(image,original_image)

    cv2.imwrite('D:/Projects/FYP/Tea Leave Image Evaluation/results/beforeFilter.jpg',original_image)
    cv2.imwrite('D:/Projects/FYP/Tea Leave Image Evaluation/results/initialClusterdImage.jpg',image)
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




img = cv2.imread('D:/Projects/FYP/Tea Leave Image Evaluation/results/orig.jpg')

#guidedKMEANS(img,2,[(100,200,0),(200,0,100)],5)
#guidedKMEANS(img,3,[(255,0,0),(0,255,0),(125,125,0)],5)
#guidedKMEANS(img,3,[(0,0,0),(125,125,125),(250,250,250)],5)
guidedKMEANS(img,6,[0,50,100,150,200,255],5)
# count=1
# for i in range(576):
#     for j in range(1152):
#         for k in range(4):
#             count+=1
# print(count)
