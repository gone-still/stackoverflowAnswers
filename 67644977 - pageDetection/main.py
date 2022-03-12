# File        :   main.py (Corner Detection via Convex Hull)
# Version     :   1.0.0
# Description :   Script that recognizes the four corners of a page under un-even brightness.
#                 Answer for: https://stackoverflow.com/questions/67644977/recognizing-corners-page-with-opencv-partialy-fails/67645614
# Date:       :   Jan 25, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   Creative Commons CC0

# imports:
import cv2
import numpy as np


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# image path
path = "D://opencvImages//"
fileName = "page01.jpg"

# Reading an image in default mode:
inputImage = cv2.imread(path + fileName)

# Deep copy for results:
inputImageCopy = inputImage.copy()

# Convert BGR to grayscale:
grayInput = cv2.cvtColor(inputImageCopy, cv2.COLOR_BGR2GRAY)

# Threshold via Otsu:
_, binaryImage = cv2.threshold(grayInput, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Get edges:
cannyImage = cv2.Canny(binaryImage, threshold1=120, threshold2=255, edges=1)

# Find the EXTERNAL contours on the binary image:
contours, hierarchy = cv2.findContours(cannyImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Store the corners:
cornerList = []

# Look for the outer bounding boxes (no children):
for i, c in enumerate(contours):

    # Approximate the contour to a polygon:
    contoursPoly = cv2.approxPolyDP(c, 3, True)

    # Convert the polygon to a bounding rectangle:
    boundRect = cv2.boundingRect(contoursPoly)

    # Get the bounding rect's data:
    rectX = boundRect[0]
    rectY = boundRect[1]
    rectWidth = boundRect[2]
    rectHeight = boundRect[3]

    # Estimate the bounding rect area:
    rectArea = rectWidth * rectHeight

    # Set a min area threshold
    minArea = 100000

    # Filter blobs by area:
    if rectArea > minArea:

        # Get the convex hull for the target contour:
        hull = cv2.convexHull(c)
        # (Optional) Draw the hull:
        color = (0, 0, 255)
        cv2.polylines(inputImageCopy, [hull], True, color, 2)

        # Create image for good features to track:
        (height, width) = cannyImage.shape[:2]
        # Black image same size as original input:
        hullImg = np.zeros((height, width), dtype=np.uint8)

        # Draw the points:
        cv2.drawContours(hullImg, [hull], 0, 255, 2)
        showImage("hullImg", hullImg)

        # Set the corner detection:
        maxCorners = 4
        qualityLevel = 0.01
        minDistance = int(max(height, width) / maxCorners)

        # Get the corners:
        corners = cv2.goodFeaturesToTrack(hullImg, maxCorners, qualityLevel, minDistance)
        corners = np.int0(corners)

        # Loop through the corner array and store/draw the corners:
        for c in corners:
            # Flat the array of corner points:
            (x, y) = c.ravel()
            # Store the corner point in the list:
            cornerList.append((x, y))

            # (Optional) Draw the corner points:
            cv2.circle(inputImageCopy, (x, y), 5, 255, 5)
            showImage("corners", inputImageCopy)
