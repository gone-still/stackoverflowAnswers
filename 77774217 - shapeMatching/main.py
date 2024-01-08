# File        :   main.py (Shape Matching)
# Version     :   1.0.0
# Description :   Script that finds a mismatched shaped in an image
#                 Answer for: https://stackoverflow.com/q/77774217/

# Date:       :   Jan 07, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   Creative Commons CC0

import numpy as np
import cv2
import math


def readImage(imagePath):
    """Reads image via OpenCV"""
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        raise TypeError("readImage>> Error: Could not load Input image.")
    return inputImage


def showImage(imageName, inputImage):
    """Shows an image in a window"""
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


def getScaledMoments(inputContour):
    """Computes log-scaled hu moments of a contour array"""
    # Calculate Moments
    moments = cv2.moments(inputContour)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)

    # Log scale hu moments
    for m in range(0, 7):
        huMoments[m] = -1 * math.copysign(1.0, huMoments[m]) * math.log10(abs(huMoments[m]))

    return huMoments


# Set image path
directoryPath = "D://opencvImages//shapes//"

imageNames = ["01", "02", "03", "04", "05"]

# Loop through the image file names:
for imageName in imageNames:

    # Set image path:
    imagePath = directoryPath + imageName + ".png"

    # Load image:
    inputImage = readImage(imagePath)
    showImage("Input Image", inputImage)

    # To grayscale:
    grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Otsu:
    binaryImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    showImage("Binary", binaryImage)

    # Contour list:
    # Store here all contours of interest (large area):
    contourList = []

    # Find the contour on the binary image:
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):

        # Get blob area:
        currentArea = cv2.contourArea(c)

        # Set min area:
        minArea = 1000

        if currentArea > minArea:
            # Approximate the contour to a polygon:
            contoursPoly = cv2.approxPolyDP(c, 3, True)

            # Get the polygon's bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)

            # Get contour centroid:
            cx = int(int(boundRect[0]) + 0.5 * int(boundRect[2]))
            cy = int(int(boundRect[1]) + 0.5 * int(boundRect[3]))

            # Store in dict:
            contourDict = {"Contour": c, "Rectangle": tuple(boundRect), "Centroid": (cx, cy)}

            # Into the list:
            contourList.append(contourDict)

    # Get total contours in the list:
    totalContours = len(contourList)

    # Deep copies of input image for results:
    inputCopy = inputImage.copy()
    contourCopy = inputImage.copy()

    # Set contour 0 as objetive:
    currentDict = contourList[0]
    # Get objective contour:
    objectiveContour = currentDict["Contour"]

    # Draw objective contour in green:
    cv2.drawContours(contourCopy, [objectiveContour], 0, (0, 255, 0), 3)

    # Draw contour index on image:
    center = currentDict["Centroid"]
    cv2.putText(contourCopy, "0", center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Store contour distances here:
    contourDistances = []

    # Calculate log-scale hu moments of objective contour:
    huMomentsObjective = getScaledMoments(objectiveContour)

    # Start from objectiveContour+1, get target contour, compute scaled moments and
    # get Euclidean distance between the two scaled arrays:

    for i in range(1, totalContours):
        # Set target contour:
        currentDict = contourList[i]
        # Get contour:
        targetContour = currentDict["Contour"]

        # Draw target contour in red:
        cv2.drawContours(contourCopy, [targetContour], 0, (0, 0, 255), 3)

        # Calculate log-scale hu moments of target contour:
        huMomentsTarget = getScaledMoments(targetContour)

        # Compute Euclidean distance between the two arrays:
        contourDistance = np.linalg.norm(np.transpose(huMomentsObjective) - np.transpose(huMomentsTarget))
        print("contourDistance:", contourDistance)

        # Store distance along contour index in distance list:
        contourDistances.append([contourDistance, i])

        # Draw contour index on image:
        center = currentDict["Centroid"]
        cv2.putText(contourCopy, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show processed contours:
        showImage("Contours", contourCopy)

    # Get maximum distance,
    # List to numpy array:
    distanceArray = np.array(contourDistances)

    # Get distance mean and std dev:
    mean = np.mean(distanceArray[:, 0:1])
    stdDev = np.std(distanceArray[:, 0:1])

    print("M:", mean, "Std:", stdDev)

    # Set contour 0 (default) as the contour that is different from the rest:
    contourIndex = 0
    minSigma = 1.0

    # If std dev from the distance array is above a minimum variation,
    # there's an outlier (max distance) in the array, thus, the real different
    # contour we are looking for:

    if stdDev > minSigma:
        # Get max distance:
        maxDistance = np.max(distanceArray[:, 0:1])
        # Set contour index (contour at index 0 was the objective!):
        contourIndex = np.argmax(distanceArray[:, 0:1]) + 1
        print("Max:", maxDistance, "Index:", contourIndex)

    # Fetch dissimilar contour, if found,
    # Get boundingRect:
    boundingRect = contourList[contourIndex]["Rectangle"]

    # Get the dimensions of the bounding rect:
    rectX = boundingRect[0]
    rectY = boundingRect[1]
    rectWidth = boundingRect[2]
    rectHeight = boundingRect[3]

    # Draw dissimilar (mismatched) contour in red:
    color = (0, 0, 255)
    cv2.rectangle(inputCopy, (int(rectX), int(rectY)),
                  (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
    showImage("Mismatch", inputCopy)
