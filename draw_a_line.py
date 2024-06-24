import cv2 as cv
import numpy as np

space = np.zeros(
    (512, 512, 3), np.uint8)
p0 = 10, 10
p1 = 300, 90
p2 = 500, 10
RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)


cv.line(space, p0, p1, RED, 2)
cv.line(space, p1, p2, GREEN, 5)
cv.line(space, p2, p0, CYAN, 9)
cv.imshow("RGB", space)

cv.waitKey(10000)
cv.destroyAllWindows()