# File        :   main.py (Biggest Blob finder)
# Version     :   1.0.0
# Description :   Script that finds and extracts the biggest binary blob
#                 Answer for: https://stackoverflow.com/q/74846184/12728244
# Date:       :   Dec 21, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   Creative Commons CC0

import numpy as np
import cv2


# Reads image via OpenCV:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)

# image path
path = "D://opencvImages//"

# Reading an image in default mode:
inputImage = readImage(path + "testBlob.png")

# Grayscale conversion:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Threshold via Otsu:
# Note the image inversion:
_, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Store a copy of the input image:
biggestBlob = binaryImage.copy()
# Set initial values for the
# largest contour:
largestArea = 0
largestContourIndex = 0

# Find the contours on the binary image:
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour in the contours list:
for i, cc in enumerate(contours):
    # Find the area of the contour:
    area = cv2.contourArea(cc)
    # Store the index of the largest contour:
    if area > largestArea:
        largestArea = area
        largestContourIndex = i

# Once we get the biggest blob, paint it black:
tempMat = binaryImage.copy()
cv2.drawContours(tempMat, contours, largestContourIndex, (0, 0, 0), -1, 8, hierarchy)

# Erase smaller blobs:
biggestBlob = biggestBlob - tempMat

# Show the result:
showImage("biggestBlob", biggestBlob)