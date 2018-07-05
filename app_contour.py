#Application of Contour
import numpy as np
import cv2
from matplotlib import pyplot
#Read the image
img=cv2.imread("rec1.jpg")
#convert color image to grayscale
gr=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#find the contours
ix,contour,hie=cv2.findContours(gr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Binary threshold
ret,thresh=cv2.threshold(gr,123,255,cv2.THRESH_BINARY)
#Find contour for binary threshold
ixx,contour1,hie1=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#Draw contours
cv2.drawContours(gr,contour1,-1,(0,255,0),3)
#Bounding box 
for c in contour1:
    x1,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(gr,(x1,y),(x1+w,y+h),(0,255,0),2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(thresh,[box],0,(255,0,255),1)
    (x,y),radius = cv2.minEnclosingCircle(c)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(gr,center,radius,(0,255,0),3)
cv2.imshow("co Image_of _contour1",gr)
print("Number of Contour:")
print(len(contour1)-1)



