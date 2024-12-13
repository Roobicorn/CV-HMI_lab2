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
number_of_images = 50
imglist = []
success = True

#Initialise empty list for chessboard corner points in world and image co-ordinates
objpoints = []
imgpoints = []
retlist = []

#Initialise chessboard object dimensions #8mm when using phone, 25mm for printed sheet
objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = 8*np.mgrid[0:6, 0:9].T.reshape(-1,2)
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
#     print("Image captured")

# #The Image index loop ends when number_of_images habe been captured
# print("Captured", capcount, "images")

# #Clean up the viewing window and release the VideoCapture instance
# cv2.destroyWindow("Captured Image")
# cv2.destroyWindow("Video Stream")
# cap.release() 
#####################################################################

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
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2],
    None, (cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 30, 0.0001))

#output and save calibration data
print("\nIntrinsic Matrix: \n")
print(mtx)
print("\nDistortion Coefficients: \n")
print(dist)
print("\nRotation Vectors:\n")
print(rvecs)
print("\nTranslation Vectors:\n")
print(tvecs)

with open("intrinsic_matrix.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(mtx)

with open("distortion_coefficients.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(dist)

with open("imgpoints.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(imgpoints)

with open("objpoints.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(objpoints)

with open("rotation_vectors.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(rvecs)

with open("translation_vectors.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(tvecs)


#Create undistorted image
img = cv2.imread("Image000.png")

#print("mtx from memory\n", mtx)
#print("dist from memory\n", dist)

mtx = np.genfromtxt("intrinsic_matrix.csv", delimiter=",")
dist = np.genfromtxt("distortion_coefficients.csv", delimiter=",")

#print("mtx from file\n", mtx)
#print("dist from file\n", dist)

#get optimal new camera mtx
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi

dst_crop = dst[y:y+h, x:x+w]
cv2.imwrite("calibrated_undistortion.png", dst)
cv2.imwrite("calibrated_undistortion_crop.png", dst_crop)

cv2.namedWindow("Original")
cv2.namedWindow("Undistorted")
cv2.namedWindow("Undistorted cropped")

cv2.imshow("Original", img)
cv2.imshow("Undistorted", dst)
cv2.imshow("Undistorted cropped", dst_crop)

cv2.waitKey(0)

cv2.destroyAllWindows()

