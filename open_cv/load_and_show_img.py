import cv2 as cv

# flower = cv.imread("flower.jpg")
# image = cv.imread("car.jpg")
image = cv.imread("background.jpg")
# cv.imshow("window", image)

cv.imwrite("background.jpg", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imwrite("background_grey.png", gray)
cv.imshow("window", gray)

cv.waitKey(30000)
cv.destroyAllWindows()