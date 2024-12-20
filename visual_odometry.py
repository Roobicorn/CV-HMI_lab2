import cv2
import numpy as np

#Create VideoCapture instance
cap = cv2.VideoCapture(1)
#Check if device can be opened, exit if not
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#Set the resolution for image capture
h = 720
w = 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

#import camera calibration parameters
mtx = np.load("intrinsic_matrix.npy")
dist = np.load("distortion_coefficients.npy")

#get optimal new camera mtx
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))
x, y, w, h = roi

#setup ORB
orb = cv2.ORB_create()

################################ Image Capture #######################
#Set up windows for the video stream and captured images
cv2.namedWindow("Captured Image")
cv2.namedWindow("Video Stream")

#Initialise an empty list of images and the number to be captured
number_of_images = 2
imglist = []
imgpoints = []
imgdescs = []
success = True

#Loop through the indices of images to be captured
for imgnum in range(number_of_images):
    #Capture images continuously and wait for a keypress
    while success and cv2.waitKey(1) == -1:
        #Read an image from the Videocapture instance
        success, img = cap.read()

        # undistort using undistort method
        img = cv2.undistort(img, mtx, dist, None, newcameramtx)
        img = img[y:y+h, x:x+w]

        #find and display ORB features
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img_orb = cv2.drawKeypoints(img, kp, None, flags=0)
        
        #Display the image
        cv2.imshow("Video Stream", img_orb)
   
    #When we exit the capture loop we save the last image and repeat
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image",imgnum,"captured")
    imgpoints.append(kp)
    imgdescs.append(des)
    
#The Image index loop ends when number_of_images habe been captured
print("Captured", len(imglist), "images")

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("odo_image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

cap.release()

######################## Feature Matching ##########################

if (len(imglist) > 1):
    # img1 = cv2.imread("odo_image001.png")
    # img2 = cv2.imread("odo_image002.png")
    img1 = imglist[-1]
    img2 = imglist[-2]
    kp1 = imgpoints[-1]
    kp2 = imgpoints[-2]
    des1 = imgdescs[-1]
    des2 = imgdescs[-2]

    #Create brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #obtain best matches between 2 sets of keypoint descriptors
    matches = bf.match(des1, des2)
    #sorts matches in order of lowest distance
    matches = sorted(matches, key=lambda x:x.distance)
    #draw matches onto image

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow("Matches")
    cv2.imshow("Matches", img_matches)
    cv2.imwrite("odometry.png", img_matches)

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    ####################### Visual Odometry ###########################

    # First unpack matches to their respective keypoint locations
    pts1 = []
    pts2 = []
    for m in matches:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    # convert to int32 format
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    #find Fundamental matrix F (transforms points from one image to another)
    # LMEDS is Least Median of Squares method. 
    # another option is RANSAC: Random Sample Concensus
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print("Fundamental Matrix:\n", F)
    np.save("odo_fundamental_matrix.npy", F)

    # Find Essential Matrix E 
    # (Extrinsic Homogeneous matrix representing translation & rotation of camera)
    # E = K'^T * F * K where K is the intrinsic matrix of the camera. 
    E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.LMEDS, prob = 0.999)
    print("Essential Matrix:\n", E)
    np.save("odo_essential_matrix.npy", E)

    # Finally estimate rotation and translation matrixes for ego-motion estimation from E
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)

    print("rotation matrix:\n", R)
    np.save("odo_rotation_matrix.npy", R)
    print("Translation matrix\n", t)
    np.save("odo_translation_matrix.npy", t)
    






