import cv2 as cv
class App:
    def __init__(self):
        img = cv.imread("background1.jpg")
        Window("window", img)

    def run(self):
        k = 0
        # ord - returns ASCII
        # this code exits when one presses "q"hbh
        while k != ord("q"):
            k = cv.waitKey(0)
            # chr - ASCII to value
            print(k, chr(k))

        cv.destroyAllWindows()

class Window:
    def __init__(self, winname, image):
        self.winname = winname
        self.image = image
        cv.imshow(winname, image)

if __name__ == "__main__":
    App().run()