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

#Initialise an empty list of images and the number to be captured
number_of_images = 3
success = True

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

#         # undistort using undistort method
#         img = cv2.undistort(img, mtx, dist, None, newcameramtx)
#         img = img[y:y+h, x:x+w]

#         #Display the image
#         cv2.imshow("Video Stream", img)
   
#     #When we exit the capture loop we save the last image and repeat
#     cv2.imwrite("Homography_Image%03d.png" % (imgnum), img)
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

# initialise arrays for image analysis
imglist = []
orblist = []
imgpoints = []
imgdescs = []

F_matrices = []
E_matrices = []
R_matrices = []
t_matrices = []
H_matrices = []

stitched = []

shapes = []

#setup ORB
orb = cv2.ORB_create()

#Create brute force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#setup image windows
cv2.namedWindow("Img -1")
cv2.namedWindow("Img -2")
cv2.namedWindow("Matches")
cv2.namedWindow("Stitched Image")
cv2.namedWindow("Blended Image")
cv2.namedWindow("Panorama")

for imgnum in range(number_of_images):
    img = cv2.imread("Homography_Image%03d.png" % (imgnum))
    if img.all() != None:
        imglist.append(img)

        #find and display ORB features
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        img_orb = cv2.drawKeypoints(img, kp, None, flags=0)
        
        orblist.append(img_orb)
        imgpoints.append(kp)
        imgdescs.append(des)

        shapes.append(img.shape[:2]) #h, w

        if len(imglist) > 1:           
            ################## Matching #########################
            #obtain best matches between 2 sets of keypoint descriptors
            matches = bf.match(imgdescs[-1], imgdescs[-2])
            #sorts matches in order of lowest distance
            matches = sorted(matches, key=lambda x:x.distance)
            #draw matches onto image
            img_matches = cv2.drawMatches(imglist[-1], imgpoints[-1], imglist[-2], imgpoints[-2], 
                matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # #K-Nearest Neighbour matching
            # # Not working - requires mask?

            # cv2.namedWindow("knnMatches")
            # ratio = 0.7
            # kmatches = []
            # knnMatches = bf.knnMatch(imgdescs[-1], imgdescs[-2], k=2)
            # for m,n in knnMatches:
            #     if m.distance < n.distance * ratio:
            #         kmatches.append(m)
            # img_kmatches = cv2.drawMatchesKnn(imglist[-1], imgpoints[-1], imglist[-2], imgpoints[-2], 
            #     kmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow("knnMatches", img_kmatches)

            cv2.imshow("Img -1", orblist[-1])
            cv2.imshow("Img -2", orblist[-2])
            cv2.imshow("Matches", img_matches)
            
            # First unpack matches to their respective keypoint locations
            pts1 = []
            pts2 = []
            for m in matches:
                pts2.append(imgpoints[-2][m.trainIdx].pt)
                pts1.append(imgpoints[-1][m.queryIdx].pt)
            # convert to int32 format
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)

            ############## Calculate Transformation Matrices ############
            #find Fundamental matrix F
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
            print("Fundamental Matrix:\n", F)
            F_matrices.append(F)

            # Find Essential Matrix E 
            E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.LMEDS, prob = 0.999)
            print("Essential Matrix:\n", E)
            E_matrices.append(E)

            # Finally estimate rotation and translation matrixes for ego-motion estimation from E
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mtx)

            print("rotation matrix:\n", R)
            R_matrices.append(R)
            print("Translation matrix\n", t)
            t_matrices.append(t)

            ######################## Stitching ########################

            H, mask = cv2.findHomography(pts1, pts2, cv2.LMEDS, 5.0)
            print("Homography Matrix:\n", H)
            H_matrices.append(H)

            warped = cv2.warpPerspective(imglist[-1], H, (imglist[-2].shape[1], imglist[-2].shape[0]))
            #stitched += warped
            stitched = imglist[-2]+warped
            
            cv2.imshow("Stitched Image", stitched)
            cv2.imwrite("StitchedPair%03d.png" % imgnum, stitched)

            ####################### Blending #########################

            alpha = 0.5
            blended = cv2.addWeighted(warped, alpha, imglist[-2], 1-alpha, 0)

            cv2.imshow("Blended Image", blended)
            cv2.imwrite("BlendedPair%03d.png" % imgnum, blended)

            ###################### Panoramas #########################

            #get dimensions of input images
            w1, h1 = shapes[-1][::-1]
            w2, h2 = shapes[-2][::-1]
            
            # warp corners of transformed image to define how they will fit in panorama
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

            warpedcorners2 = cv2.perspectiveTransform(corners2, H)

            # Define image size that includes all corners of images warped and transformed within it
            corners = np.concatenate((corners1, warpedcorners2), axis=0)
            [xmin, ymin] = np.int32(corners.min(axis = 0).ravel() - 0.5)
            [xmax, ymax] = np.int32(corners.max(axis = 0).ravel() + 0.5)

            # Define offset of one image w.r.t the other within the stitched image
            offset = [-xmin, -ymin]
            Ht = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]])

            # Warp transformed image to match frame of reference of base image
            warped1 = cv2.warpPerspective(imglist[-1], Ht @ H, (xmax - xmin, ymax - ymin))

            # "insert" unwarped image into warped one at offset defined above

            warped1[offset[1]:h1 + offset[1], offset[0]:w1 + offset[0]] = imglist[-2]

            # # YOU ARE HERE.
            # # Not sure where mask is used?
            # mask = np.where(warped1 != 0, 1, 0).astype(np.float32)
            # stitchblended = cv2.addWeighted(warped, alpha, imglist[-1], 1-alpha, 0)
            # cv2.imshow("Panorama", stitchblended)

            cv2.imshow("Panorama", cv2.resize(warped1, (0,0), fx=0.4, fy=0.4))
            cv2.imwrite("Panorama%03d.png" % imgnum, warped1)


            cv2.waitKey(0)

        else:
            stitched = imglist[0]

# print("Fs\n", F_matrices)
# print("Es\n", E_matrices)
# print("Rs\n", R_matrices)
# print("ts\n", t_matrices)
# print("Hs\n", H_matrices)

cv2.destroyWindow("Img -1")
cv2.destroyWindow("Img -2")
cv2.destroyWindow("Matches")
#cv2.destroyWindow("knnMatches")
cv2.destroyWindow("Stitched Image")
cv2.destroyWindow("Blended Image")
cv2.destroyWindow("Panorama")

################################################