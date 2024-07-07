import cv2 as cv
import numpy as np

margin = 20
rect = np.zeros((200, 200), np.uint8)
cv.rectangle(rect, (margin, margin), (200-margin, 200-margin), 150, -1)

circle = np.zeros((200, 200), np.uint8)
cv.circle(circle, (100, 100), 80, 255, -1)

cv.imshow("window", rect)
cv.waitKey(10000)
cv.destroyAllWindows()