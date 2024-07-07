import cv2 as cv

# flip an image using keys
image = cv.imread("fish.png")
cv.imshow("image", image)

while True:
    k = cv.waitKey(0)
    print(k)
    if k == ord('q'):
        break
    elif k == ord('v'):
        image = cv.flip(image, 0)
    elif k == ord('h'):
        image = cv.flip(image, 1)

    cv.imshow("image", image)

cv.destroyAllWindows()