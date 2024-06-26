import cv2 as cv
import numpy as np

space = np.zeros((512, 512, 3), np.uint8)
gray_space = np.zeros((512, 600), np.uint8)
p0 = 10, 30
p1 = 200, 200
p2 = 500, 30
RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)


cv.line(space, p0, p1, RED, 2)
cv.line(space, p1, p2, GREEN, 5)
cv.line(space, p2, p0, CYAN, 9)
cv.imshow("RGB", space)
cv.waitKey(5000)

cv.line(gray_space, p0, p1, 50, 8)
cv.line(gray_space, p1, p2, 100, 5)
cv.line(gray_space, p2, p0, 255, 2)
cv.imshow("Gray", gray_space)
cv.waitKey(5000)

cv.destroyAllWindows()