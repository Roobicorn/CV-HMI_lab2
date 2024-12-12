import cv2
import numpy as np

#Create VideoCapture instance
cap = cv2.VideoCapture(0)
#Check if device can be opened, exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Set the resolution for image capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#Set up windows for the video stream and captured images
cv2.namedWindow("Captured Image")
cv2.namedWindow("Video Stream")


#Initialise an empty list of images and the number to be captured
number_of_images = 10
imglist = []
success = True

#Loop through the indices of images to be captured
for imgnum in range(number_of_images):
    #Capture images continuously and wait for a keypress
    while success and cv2.waitKey(1) == -1:
        #Read an image from the Videocapture instance
        success, img = cap.read()
        
        #Display the image
        cv2.imshow("Video Stream", img)
   
    #When we exit the capture loop we save the last image and repeat
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image captured")

#The Image index loop ends when number_of_images habe been captured
print("Captured", len(imglist), "images")

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

cap.release()    