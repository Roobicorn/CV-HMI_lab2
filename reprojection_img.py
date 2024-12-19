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
number_of_images = 100


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

#setup display window
cv2.namedWindow("Projected Image")

#Read images from files:
for imgnum in range(number_of_images):
    #img = cv2.imread("Image%03d.png" % imgnum)
    img = cv2.imread("calibrated_undistortion_crop_%03d.png" % imgnum)

    ret, imgpoints = cv2.findChessboardCorners(img, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        img = cv2.drawChessboardCorners(img, (9,6), imgpoints, ret)
        ret, rvecs, tvecs = cv2.solvePnP(objp, imgpoints, mtx, dist)
        projpoints, jacobian = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img, imgpoints, projpoints)

        cv2.imshow("Projected Image", img)
        cv2.waitKey(0)
        
        cv2.imwrite("Reprojected_Undist_Image%03d.png" % imgnum, img)

        print("ReprojectedImage%03d.png captured" % imgnum)
    else:
        print("Chessboard not detected in Image%03d.png" % imgnum)
