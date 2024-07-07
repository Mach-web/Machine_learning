import cv2 as cv
import numpy as np

img = cv.imread('background1.jpg')

print("Img shape: {} \nImg[:2]: {}".format(img.shape, img.shape[:2]))
# get image height and width
height, width = img.shape[:2]
center = (height // 2, width // 2)
print(f"Center of Image: {center}")

'''M = cv.getRotationMatrix2D(center, -60, 0.7)
rotated = cv.warpAffine(img, M, (width, height))
cv.imshow("window", rotated)'''

def track(angle_scale):
    # track rotation trackbar
    # M = cv.getRotationMatrix2D(center, angle_scale, 1.0)
    # create a scale trackbar
    M = cv.getRotationMatrix2D(center, 0, angle_scale / 10)
    rotated_scaled = cv.warpAffine(img, M, (width, height))
    cv.imshow("window", rotated_scaled)

cv.imshow("window", img)
# create a trackbar for rotation
# cv.createTrackbar('angl', 'window', 0, 270, track)
# create trackbar for scaling
cv.createTrackbar('scale', 'window', 10, 30, track)
cv.waitKey(20000)
cv.destroyAllWindows()
