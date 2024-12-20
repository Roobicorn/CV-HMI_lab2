import cv2
import numpy as np
import csv

#Create VideoCapture instance
cap = cv2.VideoCapture(1)
#Check if device can be opened, exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Set the resolution for image capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


#Initialise an empty list of images and the number to be captured
number_of_images = 50
imglist = []
success = True

#Initialise empty list for chessboard corner points in world and image co-ordinates
objpoints = []
imgpoints = []
retlist = []

#Initialise chessboard object dimensions #8mm when using phone, 2.475cm for printed sheet
grid_unit_length = 2.475
objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = grid_unit_length*np.mgrid[0:9, 0:6].T.reshape(-1,2) #corrected 0:9, 0:6 from 0:6,0:9
#print(objp)

############ Comment out this block to avoid repeating capture ################
# #Capture images from video
# #Set up windows for the video stream and captured images
# cv2.namedWindow("Captured Image")
# cv2.namedWindow("Video Stream")

# #Loop through the indices of images to be captured
# capcount = 0
# for imgnum in range(number_of_images):
#     #Capture images continuously and wait for a keypress
#     while success and cv2.waitKey(1) == -1:
#         #Read an image from the Videocapture instance
#         success, img = cap.read()

#         #Display the image
#         cv2.imshow("Video Stream", img)
   
#     #When we exit the capture loop we save the last image and repeat
#     cv2.imwrite("Image%03d.png" % (imgnum), img)
#     capcount += 1
#     cv2.imshow("Captured Image", img)
#     print("Image", imgnum, "captured")

# #The Image index loop ends when number_of_images habe been captured
# print("Captured", capcount, "images")

# #Clean up the viewing window and release the VideoCapture instance
# cv2.destroyWindow("Captured Image")
# cv2.destroyWindow("Video Stream") 
#####################################################################

cap.release()

#Read images from files:
for imgnum in range(number_of_images):
    img = cv2.imread("Image%03d.png" % imgnum, cv2.IMREAD_GRAYSCALE)

    ret, corners = cv2.findChessboardCorners(img, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        imglist.append(img)
        objpoints.append(objp)
        imgpoints.append(corners)
        retlist.append(ret)
        print("Image%03d.png keypoints captured" % imgnum)
    else:
        print("Keypoints not detected in Image%03d.png" % imgnum)

print("Keypoints captured for", len(imglist), "images")

#Refine localisation of detected 3D points:
#print("pre refining:", imgpoints[0][0], imgpoints[5][0]) #for testing
for imgnum in range(len(imglist)):
    subcorners = cv2.cornerSubPix(imglist[imgnum], imgpoints[imgnum],(11,11), (-1,-1), 
        (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    imgpoints[imgnum] = subcorners
    print("subcorners calculated for image", imgnum)
#print("post refining:", imgpoints[0][0], imgpoints[5][0])

#calibrate camera based on data collected
print("Calibrating...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1],
    None, (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 30, 0.0001))

#output and save calibration data
print("\nIntrinsic Matrix: \n")
print(mtx)
print("\nDistortion Coefficients: \n")
print(dist)

# Save
np.save("intrinsic_matrix.npy", mtx)
np.save("distortion_coefficients.npy", dist)
