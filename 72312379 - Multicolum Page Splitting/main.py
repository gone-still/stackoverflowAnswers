# File        :   main.py (Page splitting by columns)
# Version     :   1.0.0
# Description :   Script that splits a scanned page into its columns
#                 Answer for: https://stackoverflow.com/q/72309686/12728244
# Date:       :   May 19, 2022
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


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Clamps an integer to a valid range:
def clamp(val, minval, maxval):
    if val < minval: return minval
    if val > maxval: return maxval
    return val


# Image path
path = "D://opencvImages//"
fileName = "pmALU.jpg"

# Reading an image in default mode:
inputImage = cv2.imread(path + fileName)

# To grayscale:
grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Otsu Threshold:
_, binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU)

# Show the image:
showImage("binaryImage", binaryImage)

# Get image dimensions:
(imageHeight, imageWidth) = binaryImage.shape[:2]

# Set middle ROI dimensions:
middleVertical = 0.5 * imageHeight
roiWidth = imageWidth
roiHeight = int(0.1 * imageHeight)
middleRoiVertical = 0.5 * roiHeight
roiY = int(0.5 * imageHeight - middleRoiVertical)

# Draw ROI on original image:
binaryColor = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2BGR)
cv2.rectangle(binaryColor, (0, roiY), (imageWidth, roiY + roiHeight), (0, 0, 255), 10)
# Show the image:
showImage("binaryColor", binaryColor)

# Slice the ROI:
middleRoi = binaryImage[roiY:roiY + roiHeight, 0:imageWidth]
# Show the image:
showImage("middleRoi", middleRoi)

# White rectangle around ROI:
rectangleThickness = int(0.01 * imageHeight)
cv2.rectangle(middleRoi, (0, 0), (roiWidth, roiHeight), 255, rectangleThickness)
# Show the image:
showImage("middleRoi2", middleRoi)

# Image reduction to a row:
reducedImage = cv2.reduce(middleRoi, 0, cv2.REDUCE_MIN)
# Show the image:
showImage("reducedImage", reducedImage)

# Flood fill at the extreme corners:
fillPositions = [0, imageWidth - 1]

for i in range(len(fillPositions)):
    # Get flood-fill coordinate:
    x = fillPositions[i]
    currentCorner = (x, 0)
    fillColor = 0
    cv2.floodFill(reducedImage, None, currentCorner, fillColor)
    showImage("reducedImage", reducedImage)

writeImage(path+"reducedImageFilled", reducedImage)

# Apply Opening:
kernel = np.ones((3, 3), np.uint8)
reducedImage = cv2.morphologyEx(reducedImage, cv2.MORPH_CLOSE, kernel, iterations=2)
# Show the image:
showImage("reducedImage [Morpho]", reducedImage)

# Get horizontal transitions:
whiteSpaces = np.where(np.diff(reducedImage, prepend=np.nan))[1]

# Color image:
reducedImageColor = cv2.cvtColor(reducedImage, cv2.COLOR_GRAY2BGR)
# Display the transition lines:
for y in range(len(whiteSpaces)):
    x = whiteSpaces[y]
    reducedImageColor[0, x] = (0, 255, 0)
    # Show the image:
    showImage("reducedImageColor", reducedImageColor)

# Crop the image:
colWidth = len(whiteSpaces)
spaceMargin = 0
for x in range(0, colWidth, 2):

    # Get horizontal cropping coordinates:
    if x != colWidth - 1:
        x2 = whiteSpaces[x + 1]
        spaceMargin = (whiteSpaces[x + 2] - whiteSpaces[x + 1]) // 2
    else:
        x2 = imageWidth

    # Set horizontal cropping coordinates:
    x1 = whiteSpaces[x] - spaceMargin
    x2 = x2 + spaceMargin

    print((x1, x2, spaceMargin))

    # Clamp and Crop original input:
    x1 = clamp(x1, 0, imageWidth)
    x2 = clamp(x2, 0, imageWidth)

    print((x1, x2, spaceMargin))

    currentCrop = inputImage[0:imageHeight, x1:x2]
    # Show the image:
    showImage("currentCrop", currentCrop)