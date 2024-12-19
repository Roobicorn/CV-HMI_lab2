import cv2
import numpy as np
import csv

#based on tutorial from longervision.github.io

#Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

############################## Blob Detector ###############################

#setup simpleBlobDetector parameters
blobParams = cv2.SimpleBlobDetector_Params()

#change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

#filter by area
blobParams.filterByArea = True
blobParams.minArea = 64 #adjust to suit experiment
blobParams.maxArea = 2500 #adjust to suit experiment

#Filter by circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

#filter by convexity
blobParams.filterByConvexity= True
blobParams.minInertiaRatio = 0.01

#create a detector with above parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

############################################################################

#Original blob coordinates, supposing all blobs are of z-coordinates 0
#And, the distance between every two neighbour blob circle centres is 3.7cm
circ_dist = 3.7
circ_row_length = 4
num_circ_rows = 11
num_circles = circ_row_length * num_circ_rows

objp = np.zeros((num_circles, 3), np.float32)
for circle in range(num_circles):
    if circle % 8 < 4:
        objp[circle] = (0.5*circ_dist*(circle//4),
            (circle % 4)*circ_dist,
            0)
    else:
        objp[circle] = (0.5*circ_dist*(circle//4), 
            ((circle-4)%4*circ_dist + circ_dist/2),
            0)
    
    # print("objp[{}] =".format(circle), objp[circle])

###########################################################################

#Arrays to store object points and image points from all the images.
objpoints = [] #3d point in real world space
imgpoints = [] #2d points in image plane

######################### Capture calibration frames ######################

cap = cv2.VideoCapture(0)
found = 0
number_of_images = 50

while(found < number_of_images): 
    ret, img = cap.read() #capture frame by frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints = blobDetector.detect(gray) #detect blobs

    #Draw detected blobs as red circles, this helps cv2.findCirclesGrid()
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None, flags = cv2.CALIB_CB_ASYMMETRIC_GRID) #find the circle grid

    if ret == True:
        objpoints.append(objp) # objpoints will always be the same in 3d. ie, the circle array

        corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        #Draw and display the corners
        im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
        found += 1

    cv2.imshow("img", im_with_keypoints) #display
    cv2.waitKey(2)

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print("Calibrating...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

###########################################################################

#output and save calibration data
print("\nIntrinsic Matrix: \n")
print(mtx)
print("\nDistortion Coefficients: \n")
print(dist)

with open("intrinsic_matrix.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(np.asarray(mtx).tolist())

with open("distortion_coefficients.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(np.asarray(dist).tolist())

#load save calibration data
loadedmtx = np.genfromtxt("intrinsic_matrix.csv", delimiter=",")
loadeddist = np.genfromtxt("distortion_coefficients.csv", delimiter=",")    

print("\nLoaded Intrinsic Matrix: \n")
print(loadedmtx)
print("\nLoaded Distortion Coefficients: \n")
print(loadeddist)




