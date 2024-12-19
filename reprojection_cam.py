import cv2
import numpy as np
import csv

#Create VideoCapture instance
cap = cv2.VideoCapture(0)
#Check if device can be opened, exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Set the resolution for image capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


#Initialise an empty list of images and the number to be captured
number_of_images = 10
#imglist = []
success = True

#Initialise empty list for chessboard corner points in world and image co-ordinates
#objpoints = []
imgpoints = []
#retlist = []

#Initialise chessboard object dimensions #8mm when using phone, 25mm for printed sheet
objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = 2.5*np.mgrid[0:6, 0:9].T.reshape(-1,2)
#print(objp)

#define function to draw cartesian axes
def draw(img, originpts, imgpts):
    origin = tuple(originpts[0].ravel().astype(int))
    img = cv2.line(img, origin, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

#axis matrix: 3 orthogonal vectors
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#import camera calibration parameters
mtx = np.genfromtxt("intrinsic_matrix.csv", delimiter=",")
dist = np.genfromtxt("distortion_coefficients.csv", delimiter=",")

############ Comment out this block to avoid repeating capture ################
#Capture images from video
#Set up windows for the video stream and captured images
cv2.namedWindow("Captured Image")
cv2.namedWindow("Video Stream")

#Loop through the indices of images to be captured
capcount = 0
for imgnum in range(number_of_images):
    #Capture images continuously and wait for a keypress
    while success and cv2.waitKey(1) == -1:
        #Read an image from the Videocapture instance
        success, img = cap.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, imgpoints = cv2.findChessboardCorners(gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            img = cv2.drawChessboardCorners(img, (9,6), imgpoints, ret)
            ret, rvecs, tvecs = cv2.solvePnP(objp, imgpoints, mtx, dist)
            projpoints, jacobian = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img, imgpoints, projpoints)

        #Display the image
        cv2.imshow("Video Stream", img)
   
    #When we exit the capture loop we save the last image and repeat
    cv2.imwrite("ProjectedImage%03d.png" % (imgnum), img)
    capcount += 1
    cv2.imshow("Captured Image", img)
    print("Image captured")

#The Image index loop ends when number_of_images habe been captured
print("Captured", capcount, "images")

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")
cap.release() 
5#####################################################################