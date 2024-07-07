import cv2 as cv
import numpy as np

image = cv.imread("fish.png")
image = cv.resize(image, None, fx = 0.5, fy = 0.5, interpolation=cv.INTER_CUBIC)

mask = np.zeros(image.shape[:2], np.uint8)
cv.circle(mask, (105, 235), 80, 255, -1)

mask = cv.bitwise_and(image, image, mask=mask)
image2 = np.hstack([image, mask])

cv.imshow("window", image2)
cv.waitKey(10000)
cv.destroyAllWindows()