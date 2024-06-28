import cv2 as cv
import numpy as np

BLUE = (255, 0, 0)
center = 200, 50
axes = 100, 30
angle = 20

img = np.zeros((360, 640, 3), np.uint8)
cv.ellipse(img, center, axes, angle, 0, 360, (100, 150, 45), 2)
cv.imshow("window", img)

cv.waitKey(10000)
cv.destroyAllWindows()