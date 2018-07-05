#Threshold OpenCV
import numpy as np
import cv2
from matplotlib import pyplot
#Read the image
xx=cv2.imread("image.jpeg")
#Convert color image to grayscale
gg=cv2.cvtColor(xx,cv2.COLOR_RGB2GRAY)
#Convert to threshold values
ret,thresh=cv2.threshold(gg,123,255,cv2.THRESH_BINARY)
ret,thresh1=cv2.threshold(gg,123,255,cv2.THRESH_BINARY_INV)
ret,thresh2=cv2.threshold(gg,123,255,cv2.THRESH_TRUNC)
ret,thresh3=cv2.threshold(gg,123,255,cv2.THRESH_TOZERO)
ret,thresh4=cv2.threshold(gg,123,255,cv2.THRESH_TOZERO_INV)
img=[gg,thresh,thresh1,thresh2,thresh3,thresh4]
#Display image as subplot
for i in range(0,6):
	pyplot.subplot(2,3,i+1)
	pyplot.imshow(img[i],'gray')
pyplot.show()
