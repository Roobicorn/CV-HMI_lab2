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

################################ Image Capture #######################
#Set up windows for the video stream and captured images
cv2.namedWindow("Captured Image")
cv2.namedWindow("Video Stream")

#Initialise an empty list of images and the number to be captured
number_of_images = 2
imglist = []
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
        
        #Display the image
        cv2.imshow("Video Stream", img)
   
    #When we exit the capture loop we save the last image and repeat
    imglist.append(img)
    cv2.imshow("Captured Image", img)
    print("Image",imgnum,"captured")

#The Image index loop ends when number_of_images habe been captured
print("Captured", len(imglist), "images")

#Save all images to image files for later use
for imgnum, img in enumerate(imglist):
    cv2.imwrite("Disparity_Image%03d.png" % (imgnum), img)

#Clean up the viewing window and release the VideoCapture instance
cv2.destroyWindow("Captured Image")
cv2.destroyWindow("Video Stream")

cap.release()

######################## Disparity Map ##########################

if True: #(len(imglist) > 1):
    #convert to greyscale
    gray1 = cv2.imread("Disparity_Image000.png", cv2.IMREAD_GRAYSCALE)
    gray2 = cv2.imread("Disparity_Image001.png", cv2.IMREAD_GRAYSCALE)

    #create stereo block matcher with default parameters
    stereo = cv2.StereoBM.create(numDisparities=64, blockSize=5)
    stereo.setTextureThreshold(0)
    stereo.setUniquenessRatio(5)
    stereo.setSpeckleWindowSize(0)


    #compute disparity map
    disparity = stereo.compute(gray1, gray2)

    disparity = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.namedWindow("Disparity Map")
    cv2.imshow("Disparity Map", disparity)

    cv2.imwrite("Disparity_map.png", disparity)

    cv2.waitKey(0)
