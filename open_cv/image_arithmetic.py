import cv2 as cv
import numpy as np

image = cv.imread('fish.png')
image = cv.resize(image, None, fx = 0.4, fy = 0.4, interpolation=cv.INTER_CUBIC)
M = np.ones(image.shape, dtype = np.uint8)
M *= 100

brighter = cv.add(image, M)
darker = cv.subtract(image, M)

img2 = np.hstack([image, brighter, darker])
cv.imshow("window", img2)

cv.waitKey(10000)
cv.destroyAllWindows()