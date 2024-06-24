import cv2 as cv
import numpy as np

# create a 3D array of zeros
img = np.ones((300, 600,3), np.uint8)
cv.imshow("RGB", img)

# Display gray scale zeros
gray_scale = np.ones((300, 600, 3), np.uint8)
cv.imshow("Gray", gray_scale)

cv.waitKey(10000)
cv.destroyAllWindows()