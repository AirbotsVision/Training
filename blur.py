import cv2
import numpy as np
import matplotlib
import imutils
import glob

b=0
c=0
d=0
e=0

xx=[]
yy=[]
ww=[]
zz=[]
#filenames = glob.glob('C:\\Users\Airbots\Desktop\Python basic\All_Image_and_Kernels/*.jpg')

for img in glob.glob("All_Image_and_Kernels/*.jpg"):
      x=img
      img=cv2.imread(img)
      #print(img))
      gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

      #fm = variance_of_laplacian(gray)
      blur_map =  cv2.Laplacian(gray, cv2.CV_64F).var()
      threshold=[60,70,80,90,100,110,120,140,160,180,200]
      #cv2.imshow("Imgae",img)
      for i in threshold:  
        if(blur_map<=i):
           #print("Camera Blurred")
           if 'blur' in x:
              b=b+1
           if 'blur' not in x:
              c=c+1
        else:
           if 'blur' in x:
              d=d+1
           if 'blur' not in x:
              e=e+1
        xx.append(b)
        yy.append(c)
        ww.append(d)
        zz.append(e)
print(xx,yy,ww,zz)
#print(threshold)

             
