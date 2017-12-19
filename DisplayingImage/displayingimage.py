import numpy as np 
import cv2

#Reading an image
img = cv2.imread("color.jpg",1)

#Displaying an image
cv2.imshow('Display',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
