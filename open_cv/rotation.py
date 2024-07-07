import cv2 as cv
import numpy as np

img = cv.imread('car_gray.tiff')

print("Img shape: {} \nImg[:2]: {}".format(img.shape, img.shape[:2]))
# get image height and width
height, width = img.shape[:2]
center = (height // 2, width // 2)
print(f"Center of Image: {center}")

def trackbar(angle):

