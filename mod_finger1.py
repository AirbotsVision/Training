import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
contour_length=[]
people_count=[]
def detection():
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_lower.xml')
    ori="http://85.93.105.195:8081/mjpg/video.mjpg?COUNTER"
    cap = cv2.VideoCapture(ori)

    file_name="Contour.txt"
    file_name="people.txt"
    #Open Camera object
    #cap = cv2.VideoCapture(0)

    #size=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #print(size)
    #Decrease frame size
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,900) 
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 752)
    #cap.set(cv2.CAP_PROP_SATURATION,0.2)

    '''# Creating a window for HSV track bars
    cv2.namedWindow('HSV_TrackBar')

    # Starting with 100's to prevent error while masking
    h,s,v = 100,100,100

    # Creating track bar
    cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
    cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
    cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)'''

    #contour_length=[]
    #people_count=[]
    c=0
    i=1
    while (1):
        #obj=open("Contour.txt","w")
        #obj1=open("people.txt","w")

        #Measure execution time 
        start_time = time.time()
    
        #Capture frames from the camera
        ret, frame = cap.read()
    
        #Blur the image
        #blur = cv2.medianBlur(frame,5)
        blur = cv2.blur(frame,(5,5))
 	
 	#Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        #cv2.imshow("img1",hsv)
    
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([2,50,48]),np.array([12,210,200]))
        #cv2.imshow("img",mask2)
    
    
        #Kernel matrices for morphological transformation    
        kernel_square = np.zeros((11,11),np.uint8)
        #print(kernel_square)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #print(kernel_ellipse)
        #Perform morphological transformations to filter out the background noise
        #Dilation increase skin color area
        #Erosion increase skin color area
        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        #print(dilation)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,0,255,0)
        cv2.imshow("img",thresh)
    
        #Find contours of the filtered frame
        im,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        #print(type(contours))
        '''faces = face_cascade.detectMultiScale(im,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,h+y),(0,0,255),3)
            roi_gray = im[y:y+h, x:x+w]
            roi_color = im[y:y+h, x:x+w]'''
        #Draw Contours
        #cv2.drawContours(frame, contours, -1, (122,122,0), 3)
        #print("Values:")
    
        #cv2.imshow('Dilation',median)
    
	#Find Max contour area (Assume that hand is in the frame)
        max_area=100
        ci=0	

        for i in range(0,len(contours)):
            cnt=contours[i]
            #print(len(contours))
            #Find convex hull
            hull = cv2.convexHull(cnt)
            #print(len(hull))
        
    
            #Find convex defects
            hull2 = cv2.convexHull(cnt,returnPoints = False)
            cv2.convexityDefects(cnt,hull2)
            #print(defects.shape)
            #Get fingertip points from contour hull
            #If points are in proximity of 80 pixels, consider as a single point in the group
            finger = []
            for i in range(0,len(hull)-1):
               if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) >80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1])>80):
                    if hull[i][0][1] <500:
                        finger.append(hull[i][0])
    
            #The fingertip points are 5 hull points with largest y coordinates  
            finger =  sorted(finger,key=lambda x: x[1])
            fingers = finger[0:5]
            #printnt(fingers)
            #c=c+1
        
            #Print bounding rectangle
            x,y,w,h = cv2.boundingRect(cnt)
        
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        
            #cv2.drawContours(frame,[hull],-1,(0,255,0),2)
            #m=len(cnt)-1
    
            #print(m)
        
            #print("Length")
            #print(len(contour_length))
            #print(c)
        m=len(contours)
        #obj.writeline(str(m))
        contour_length.append(m)    
        inp=input()
        people_count.append(inp)
        #obj1.writeline(str(inp))
        c=c+1
    
        ##### Show final image ########
        '''for i in range(0,len(contour_length)):
           print(contour_length[i])
        for ii in range(0,len(people_count)):
           print(people_count[ii])       
        #out=int((max-8)/2)
        #print(len(contour_length))
        #print(out)           
           #if contour_length==28:
              # print("Number of people"+"3")
           #elif contour_length==35:
              # print( "Number Of people 2")
        #for i in range(0,len(people_count)):
         #       print(people_count[i])'''      
        cv2.imshow('Dilation',frame)
        ###############################
    
        #Print execution time
        #print time.time()-start_time
        #print("1")
        #close the output video by pressing 'ESC'
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
             break
        if(c==50):
           break   
    for i in range(0,len(contour_length)):
        print(contour_length[i])
    for i in range(0,len(people_count)):
        print(people_count[i])
    cap.release()    
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x)
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return(b_0, b_1)
 
def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
 
    # predicted response vector
    y_pred = b[0] + b[1]*x
 
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
 
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
 
    # function to show plot
    plt.show()
 
def main():
   #detection
    detection()

     # observations
    y= np.asarray(contour_length).astype(np.int)
    x = np.asarray(people_count).astype(np.int)
    print(x)
    

 
    # estimating coefficients
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
 
    # plotting regression line
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()        
        
cv2.destroyAllWindows()
