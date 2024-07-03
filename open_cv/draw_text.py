import cv2 as cv
import numpy as np

RED = (0, 0, 255)
p0 = (10, 100)

font = cv.FONT_HERSHEY_SIMPLEX
img = np.zeros((200, 500, 3), np.uint8)

cv.putText(img, "Hello World", p0, font, 2, RED, 3, cv.LINE_AA)
cv.imshow("window", img)

cv.waitKey(10000)
cv.destroyAllWindows()