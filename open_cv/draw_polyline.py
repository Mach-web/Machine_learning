import cv2 as cv
import numpy as np

img = np.zeros((400, 600, 3), np.uint8)
pts = [(50, 50), (300, 190), (400, 10), (400, 400)]
cv.polylines(img, np.array([pts]), True, (34, 67, 73), 5)

cv.imshow("window", img)

cv.waitKey(10000)
cv.destroyAllWindows()