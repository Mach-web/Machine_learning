import cv2 as cv
import numpy as np
# img = np.zeros((512,512,3), np.uint8)

p0, p1 = (100, 30), (400, 90)
RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)

def mouse_callback(event, x, y, flags, param):
    if flags == 1:
        p1 = x, y
        print(event)
        # cv.displayOverlay("window", f"x: {x} y: {y}")
        img[:] = 0
        cv.line(img, p0, p1, GREEN, 2)
        cv.imshow("window", img)

def mouse(event, x, y, flags, param):
    global p0, p1
    if event == cv.EVENT_LBUTTONDOWN:
        p0 = x, y
        p1 = x, y
    elif event == cv.EVENT_MOUSEMOVE and flags == 1:
        p1 = x, y
        img[:] = img0
    elif event == cv.EVENT_LBUTTONUP:
        p1 = x, y
        img0[:] = img

    cv.line(img, p0, p1, RED, 2)
    cv.imshow("window", img)
    print("p0 = {} p1 = {}".format(p0, p1))

img0 = np.zeros((512,512,3), np.uint8)
img = img0.copy()
# cv.line(img, p0, p1, RED, 2)
cv.imshow("window", img)
cv.setMouseCallback("window", mouse)

cv.waitKey(50000)
cv.destroyAllWindows()
