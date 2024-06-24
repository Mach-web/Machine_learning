import cv2 as cv

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("dura.mp4")
while True:
    # capture frame by frame
    ret, frame = cap.read()

    # convert frame to gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow("window", gray)

    # display text
    # cv.displayOverlay("window", "Now playing:  dura\n By: Daddy Yankee", delayms=30000)

    # close window
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
