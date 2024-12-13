import numpy as np
import cv2

#Create undistorted image
img = cv2.imread("Image000.png")

mtx = np.genfromtxt("intrinsic_matrix.csv", delimiter=",")
dist = np.genfromtxt("distortion_coefficients.csv", delimiter=",")

#get optimal new camera mtx
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w,h))

# undistort using remapping method
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
x, y, w, h = roi
dst_crop = dst[y:y+h, x:x+w]

cv2.imwrite("calibrated_remapping.png", dst)
cv2.imwrite("calibrated_remapping_crop.png", dst_crop)

#display new images
cv2.namedWindow("Original")
cv2.namedWindow("Undistorted")
cv2.namedWindow("Undistorted cropped")

cv2.imshow("Original", img)
cv2.imshow("Undistorted", dst)
cv2.imshow("Undistorted cropped", dst_crop)

cv2.waitKey(0)

cv2.destroyAllWindows()