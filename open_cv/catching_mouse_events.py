import cv2 as cv

def mouse(event, x, y, flags, param):
    text = f"Event: {event}: Mouse at: ({x}, {y}) flags: {flags} Params: {param}"
    print(text)

image = cv.imread("background1.jpg")
cv.imshow("win", image)
cv.setMouseCallback("win", mouse)

cv.waitKey(30000)
cv.destroyAllWindows()