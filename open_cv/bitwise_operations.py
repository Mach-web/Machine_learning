import cv2 as cv
import numpy as np

margin = 30
rect = np.zeros((200, 200), np.uint8)
cv.rectangle(rect, (margin, margin), (200-margin, 200-margin), 255, -1)

circle = np.zeros((200, 200), np.uint8)
cv.circle(circle, (100, 100), 80, 255, -1)

bit_and = cv.bitwise_and(rect, circle)
bit_or = cv.bitwise_or(rect, circle)
bit_xor = cv.bitwise_xor(rect, circle)

img = np.hstack([rect, circle, bit_and, bit_or, bit_xor])

cv.imshow("window", img)
cv.waitKey(0000)
cv.destroyAllWindows()