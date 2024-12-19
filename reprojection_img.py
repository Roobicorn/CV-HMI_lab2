import cv2
import numpy as np
import csv

#Initialise an empty list of images and the number to be captured
number_of_images = 20

#Initialise chessboard object dimensions #8mm when using phone, 24.75mm for printed sheet
grid_unit_length = 0.8
objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = grid_unit_length*np.mgrid[0:9, 0:6].T.reshape(-1,2)
#print(objp)

#set up criteria for cornerSubPix refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#define function to draw cartesian axes
def draw(img, originpts, imgpts):
    origin = tuple(originpts[0].ravel().astype(int))
    imgpts = imgpts.reshape(-1, 2).astype(int)  # Ensure proper format
    img = cv2.line(img, origin, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
    return img

#axis matrix: 3 orthogonal vectors
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

#import camera calibration parameters
mtx = np.load("intrinsic_matrix.npy")
dist = np.load("distortion_coefficients.npy")

#setup display window
cv2.namedWindow("Projected Image")

#Read images from files:
for imgnum in range(number_of_images):
    #img = cv2.imread("Image%03d.png" % imgnum)
    img = cv2.imread("calibrated_undistortion_crop_%03d.png" % imgnum)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_width, imheight = gray.shape[::-1]

    ret, imgpoints = cv2.findChessboardCorners(gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        imgpoints = cv2.cornerSubPix(gray, imgpoints, (11,11), (-1,-1), criteria)

        img = cv2.drawChessboardCorners(img, (9,6), imgpoints, ret)
        ret, rvecs, tvecs = cv2.solvePnP(objp, imgpoints, mtx, dist)
        projpoints, jacobian = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, imgpoints, projpoints)

        cv2.imshow("Projected Image", img)
        cv2.waitKey(1)
        
        cv2.imwrite("Reprojected_undistorted_Image%03d.png" % imgnum, img)

        print("ReprojectedImage%03d.png captured" % imgnum)
    else:
        print("Chessboard not detected in Image%03d.png" % imgnum)
