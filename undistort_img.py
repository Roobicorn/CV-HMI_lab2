import numpy as np
import cv2

#import camera calibration parameters
mtx = np.load("intrinsic_matrix.npy")
dist = np.load("distortion_coefficients.npy")

#get optimal new camera mtx
img = cv2.imread("Image000.png")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))

number_of_images = 50

#Create undistorted image
for imgnum in range(number_of_images):
    img = cv2.imread("Image%03d.png" % imgnum)

    # undistort using undistort method
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst_crop = dst[y:y+h, x:x+w]

    cv2.imwrite("calibrated_undistortion_%03d.png" % imgnum, dst)
    cv2.imwrite("calibrated_undistortion_crop_%03d.png" % imgnum, dst_crop)

    #display new images
    cv2.namedWindow("Original")
    cv2.namedWindow("Undistorted")
    cv2.namedWindow("Undistorted cropped")

    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", dst)
    cv2.imshow("Undistorted cropped", dst_crop)

    cv2.waitKey(0)

cv2.destroyAllWindows()