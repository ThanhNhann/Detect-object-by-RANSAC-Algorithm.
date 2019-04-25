
# coding: utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


sift = cv2.xfeatures2d.SIFT_create()

## Create flann matcher
FLANN_INDEX_KDTREE = 1  
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#matcher = cv2.FlannBasedMatcher_create()
matcher = cv2.FlannBasedMatcher(flann_params, {})

## Detect and compute
img1 = cv2.imread("img1.pgm")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kpts1, descs1 = sift.detectAndCompute(gray1,None)

## As up
img2 = cv2.imread("img2.pgm")
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kpts2, descs2 = sift.detectAndCompute(gray2,None)

## Ratio test
point1, point2 = list(),list()
matches = matcher.knnMatch(descs1, descs2, 2)
for i, (m1,m2) in enumerate(matches):
    if m1.distance < 0.5 * m2.distance:
        
        point1.append(np.array([kpts1[m1.queryIdx].pt[0],kpts1[m1.queryIdx].pt[1]]))
        #point1.append(kpts1[m1.trainIdx].pt)

        point2.append(np.array([kpts2[m1.trainIdx].pt[0],kpts2[m1.trainIdx].pt[1]]))
        #point2.append(kpts2[m1.trainIdx].pt)

#Conver type point      
point1 = np.asarray(point1, dtype=np.float32)          
point2 = np.asarray(point2, dtype=np.float32) 
#Add column 1
point1 = np.hstack((point1,np.ones((point1.shape[0],1))))
point2 = np.hstack((point2,np.ones((point2.shape[0],1))))



#Initialize variable
threshold = 5
inliner = 0
inliner_tmp = 0
H = np.array([[0,0,0],
              [0,1,0],
              [0,0,0]])

#Loop to find the Homography matrix
for i in range(10000):
    #Get three point
    tmp =random.sample(range(point1.shape[0]),3)
    a = np.array([point1[tmp[0]],point1[tmp[1]],point1[tmp[2]]])
    b = np.array([point2[tmp[0]],point2[tmp[1]],point2[tmp[2]]])
    #Get H_tmp
    
    Htmp = (a.T).dot(np.linalg.inv(b.T))
    #Check H_tmp
    for i in range(point1.shape[0]):
        c = np.array([[point2[i][0],point2[i][1],1]])
        flag = Htmp.dot(c.T)
       
        if (flag[0]-point1[i,0])**2+(flag[1]-point1[i,1])**2 <= threshold**2:
            inliner_tmp += 1
    if inliner_tmp > inliner:
        inliner = inliner_tmp
        H = Htmp
    inliner_tmp = 0    
    
    


#Get corner
lt = (H).dot(np.array([[0,0,1]]).T)
rt = (H).dot(np.array([[img2.shape[1],0,1]]).T)
rb = (H).dot(np.array([[img2.shape[1],img2.shape[0],1]]).T)
lb = (H).dot(np.array([[0,img2.shape[0],1]]).T)




#Draw boudingbox
cv2.line(img1,(int(lt[0]),int(lt[1])),(int(rt[0]),int(rt[1])),(0,255,0))
cv2.line(img1,(int(rt[0]),int(rt[1])),(int(rb[0]),int(rb[1])),(0,255,0))
cv2.line(img1,(int(rb[0]),int(rb[1])),(int(lb[0]),int(lb[1])),(0,255,0))
cv2.line(img1,(int(lb[0]),int(lb[1])),(int(lt[0]),int(lt[1])),(0,255,0))

cv2.imwrite('Result.jpg',img1)


