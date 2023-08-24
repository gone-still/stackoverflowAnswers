# File        :   main.py (Remove Fingerprint Borders)
# Version     :   1.0.0
# Description :   Script that finds and deletes the external borders of a
#                 fingerprint image.
#                 Answer for: https://stackoverflow.com/q/76959316/12728244

# Date:       :   Aug 23, 2023
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
        raise TypeError("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Set image path
imagePath = "D://opencvImages//TIMu6.jpg"

# Load image:
image = cv2.imread(imagePath)

# Get binary image:
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binaryImage = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
showImage("Binary", binaryImage)

# BGR of binary image:
bgrImage = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
bgrCopy = bgrImage.copy()

# Get image dimensions:
imageHeight, imageWidth = binaryImage.shape[0:2]

# Vertically divide in 4 parts:
heightDivision = 4
heightPortion = imageHeight // heightDivision

# Store divisions here:
imageDivisions = []

# Check out divisions:
for i in range(heightDivision):
    # Compute y coordinate:
    y = i * heightPortion

    # Set crop dimensions:
    x = 0
    w = imageWidth
    h = heightPortion

    # Crop portion:
    portionCrop = binaryImage[y:y + h, x:x + w]

    # Store portion:
    imageDivisions.append(portionCrop)

    # Draw rectangle:
    cv2.rectangle(bgrImage, (0, y), (w, y + h), (0, 255, 0), 1)
    showImage("Portions", bgrImage)

# Reduce first portion to a row:
reducedImage = cv2.reduce(imageDivisions[0], 0, cv2.REDUCE_MAX)
showImage("Reduced Image", reducedImage)

# Get first and last white pixel positions:
pastPixel = 0
pixelCoordinates = []
for i in range(imageWidth):
    # Get current pixel:
    currentPixel = reducedImage[0][i]

    # Search for first transition black to white:
    if currentPixel == 255 and pastPixel == 0:
        pixelCoordinates.append(i)
        print("First", i)
    else:
        # Search for last transition white to black:
        if currentPixel == 0 and pastPixel == 255:
            pixelCoordinates.append(i - 1)
            print("Last", i - 1)

    # Set last pixel:
    pastPixel = currentPixel

# Flood fill original image:
color = (0, 0, 255)  # Red

for i in range(len(pixelCoordinates)):
    # Get x coordinate:
    x = pixelCoordinates[i]
    # Set y coordinate:
    y = heightPortion

    # Set seed point:
    seedPoint = (x, y)
    # Flood-fill:
    cv2.floodFill(bgrCopy, None, seedPoint, color)
    showImage("Filled", bgrCopy)
