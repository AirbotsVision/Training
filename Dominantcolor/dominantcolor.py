import numpy as np 
import cv2
from matplotlib import pyplot as plt
from PIL import Image


# Load the image
image = cv2.imread("color.jpg")

#create an image object and open the image in read mode
img = Image.open("color.jpg",'r')

#Extracting the pixel values using getdata()
pix_val = list(img.getdata())


#Counting the dulpicate keys and counting is storing as a value in dictionary
frequency = {}
for i in pix_val:
	frequency[i] = frequency.get(i, 0) + 1

#Finding key with the maximum value 
key_max = max(frequency, key=frequency.get)

#printing key and value
print key_max,frequency[key_max]
