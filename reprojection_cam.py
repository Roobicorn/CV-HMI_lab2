import cv2
import numpy as np
import csv

#Create VideoCapture instance
cap = cv2.VideoCapture(1)
#Check if device can be opened, exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

h = 720
w = 1280

#Set the resolution for image capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

#Initialise an empty list of images and the number to be captured
number_of_images = 10
success = True

#Initialise empty list for chessboard corner points in world and image co-ordinates
#objpoints = []
#imgpoints = []
#retlist = []
rvecs = []
tvecs = []

#Initialise chessboard object dimensions #8mm when using phone, 25mm for printed sheet
grid_unit_length = 2.475
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
axis = np.float32([[6,0,0], [0,6,0], [0,0,-6]]).reshape(-1,3)

#import camera calibration parameters
mtx = np.load("intrinsic_matrix.npy")
dist = np.load("distortion_coefficients.npy")

#get optimal new camera mtx
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))
x, y, w, h = roi

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

        # undistort using undistort method
        img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        img = img[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, imgpoints = cv2.findChessboardCorners(gray, (9,6), cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            imgpoints = cv2.cornerSubPix(gray, imgpoints, (11,11), (-1,-1), criteria)

            img = cv2.drawChessboardCorners(img, (9,6), imgpoints, ret)
            ret, rvec, tvec = cv2.solvePnP(objp, imgpoints, mtx, dist)
            projpoints, jacobian = cv2.projectPoints(axis, rvec, tvec, mtx, dist)

            img = draw(img, imgpoints, projpoints)

        #Display the image
        cv2.imshow("Video Stream", img)
   
    #When we exit the capture loop we save the last image and repeat
    cv2.imwrite("ProjectedImage%03d.png" % (imgnum), img)
    capcount += 1
    cv2.imshow("Captured Image", img)
    print("Image captured")

    R, _ = cv2.Rodrigues(rvec)
    print("Rotation matrix:\n", R)
    print("Translation vector:\n", tvec)
    rvecs.append(R)
    tvecs.append(tvec)

#The Image index loop ends when number_of_images habe been captured
print("Captured", capcount, "images")

# save rotation and translation vectors of camera wrt centre of chessboard for each image
with open("rotation_vectors.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(rvecs)

with open("translation_vectors.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(tvecs)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")
cap.release() 
#####################################################################