import cv2 as cv

# flower = cv.imread("flower.jpg")
# image = cv.imread("car.jpg")
image = cv.imread("background.jpg")
cv.imshow("window", image)
cv.waitKey(10000)
cv.destroyAllWindows()