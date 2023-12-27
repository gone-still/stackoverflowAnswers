# File        :   main.py (Color Cell Location)
# Version     :   0.5.0
# Description :   Script that locates color cells from a color card picture.
#                 Partial Answer for: ???

# Date:       :   Dec 26, 2023
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


# Gets a list of bounding rectangles and creates a composite
# bounding rectangle:
def getCombinedRectangle(rectangleList):
    # Get top left point:
    rectMatrix = np.array(rectangleList)
    compX = np.min(rectMatrix[:, 0])
    compY = np.min(rectMatrix[:, 1])

    # Get bottom right point:
    compW = 0
    compH = 0

    maxWidth = 0
    maxHeight = 0

    for currentRect in rectangleList:
        # Get coordinates:
        sx = currentRect[0]
        sy = currentRect[1]
        ex = (sx + currentRect[2]) - compX
        ey = (sy + currentRect[3]) - compY

        # Keep the largest width:
        if ex > maxWidth:
            compW = ex
            maxWidth = ex

        # Keep the largest height:
        if ey > maxHeight:
            compH = ey
            maxHeight = ey

    # Ready:
    return compX, compY, compW, compH


# Applies k-means for image segmentation:
def imageQuantization(inputImage, k, runDistanceFilter, filterParams, minDistance):
    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixelValues = inputImage.reshape((-1, 3))
    # Convert to float
    pixelValues = np.float32(pixelValues)

    # Define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)

    # Run k-means, get cluster centers:
    _, labels, (centers) = cv2.kmeans(pixelValues, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back to 8 bit values:
    centers = np.uint8(centers)

    # Run distance filter:
    if runDistanceFilter:
        # Loop through the centers array and look for the filter targets
        # Apply colors according to the dict params:
        for i in range(k):
            # Get current center:
            currentPixel = centers[i]
            # Get dict lower and upper thresholds:
            targetColor = filterParams["Threshold"]
            subColor = filterParams["Low"]
            # Check if current center is within threshold range:
            for currentTarget in targetColor:
                currentDistance = np.linalg.norm(currentPixel - currentTarget)
                if currentDistance < minDistance:
                    subColor = filterParams["High"]
            # Default value:
            centers[i] = subColor

    # Flatten the labels array:
    labels = labels.flatten()

    # Convert all pixels to the color of the centroids:
    segmentedImage = centers[labels.flatten()]
    # Reshape back to the original image dimension
    segmentedImage = segmentedImage.reshape(currentCrop.shape)

    # Ready:
    return segmentedImage


# Set image path
directoryPath = "D://opencvImages//"

# Set image names:
imagesList = ["chart-01.jpg", "chart-02.jpg", "chart-03.jpg"]

for currentImage in imagesList:

    # Set image path:
    imagePath = directoryPath + currentImage

    # Load image:
    inputImage = cv2.imread(imagePath)

    # showImage("Input Image", inputImage)

    # Deep copy for results:
    inputImageCopy = inputImage.copy()

    # Get local maximum:
    kernelSize = 100
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    localMax = cv2.morphologyEx(inputImage, cv2.MORPH_CLOSE, maxKernel, None, None, 1, cv2.BORDER_REFLECT101)

    # Perform gain division
    gainDivision = np.where(localMax == 0, 0, (inputImage / localMax))

    # Clip the values to [0,255]
    gainDivision = np.clip((255 * gainDivision), 0, 255)

    # Convert the mat type from float to uint8:
    gainDivision = gainDivision.astype("uint8")

    # showImage("Gain Div", gainDivision)

    # Convert the BGR pixel to HSV:
    hsvImage = cv2.cvtColor(gainDivision, cv2.COLOR_BGR2HSV)

    # Threshold on red:
    rangeThreshold = 5
    lowerValues = np.array([159, 70, 109])
    upperValues = np.array([179, 255, 255])

    # Create HSV mask:
    redMask = cv2.inRange(hsvImage, lowerValues, upperValues)
    # showImage("Red Mask", redMask)

    # Use a little bit of morphology to clean the mask:
    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 5

    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    redMask = cv2.morphologyEx(redMask, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations,
                               cv2.BORDER_REFLECT101)

    # Deep copy for results:
    binaryColor = redMask.copy()
    binaryColor = cv2.cvtColor(binaryColor, cv2.COLOR_GRAY2BGR)

    # Get the center of the image:
    imageHeight, imageWidth = redMask.shape[:2]
    imageCx = imageWidth // 2
    imageCy = imageHeight // 2

    color = (0, 255, 0)
    cv2.circle(binaryColor, (imageCx, imageCy), 10, color, -1)

    # Find the blobs on the binary image:
    contours, hierarchy = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Store the bounding rects here:
    boundingRects = {"Strip": [], "Card": []}

    # Set a min area threshold:
    minArea = 2000

    # Look for the outer bounding boxes (no children):
    for i, c in enumerate(contours):

        # Get blob area:
        currentArea = cv2.contourArea(c)
        print("fuck", currentArea, "h", hierarchy[0][i])

        if currentArea > minArea:

            # Approximate the contour to a polygon:
            contoursPoly = cv2.approxPolyDP(c, 3, True)
            # Get the polygon's bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)

            # Get the dimensions of the bounding rect:
            rectX = boundRect[0]
            rectY = boundRect[1]
            rectWidth = boundRect[2]
            rectHeight = boundRect[3]

            # Get contour centroid:
            contourCx = rectX + rectWidth // 2
            contourCy = rectY + rectHeight // 2

            # Get aspect ratio:
            aspectRatio = rectHeight / rectWidth
            print(aspectRatio)

            # Target contours have aspectRatio > 1
            if aspectRatio > 1:

                # Draw centroid:
                color = (255, 0, 0)
                cv2.circle(binaryColor, (contourCx, contourCy), 5, color, -1)

                # Strip should be on the left:
                if contourCx < imageCx:
                    # Collect all the rectangles that belong to the strip:
                    currentShape = "Strip"
                    # Set bounding rect:
                    color = (0, 255, 0)
                else:
                    # Card:
                    currentShape = "Card"
                    # Set bounding rect:
                    color = (0, 0, 255)

                # Store in dictionary of rectangles:
                boundingRects[currentShape].append(list(boundRect))
                print(boundingRects)

                cv2.rectangle(binaryColor, (int(rectX), int(rectY)),
                              (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

            # showImage("Rectangles", binaryColor)

    # Get combined bounding rectangle for the strip:
    stripX, stripY, stripWidth, stripHeight = getCombinedRectangle(boundingRects["Strip"])

    cv2.rectangle(inputImageCopy, (stripX, stripY), (stripX + stripWidth, stripY + stripHeight), (255, 0, 0), 3)
    # showImage("Strip Rect", inputImageCopy)

    # Update strip rectangle:
    boundingRects["Strip"] = [stripX, stripY, stripWidth, stripHeight]
    # Card list should be one element:
    if len(boundingRects["Card"]) > 1:
        raise "Card list has more than one element! (You may want to combine rects)."
    else:
        # "Flatten" card list:
        boundingRects["Card"] = boundingRects["Card"][0]

    shapeParams = {"Strip": {"Clusters": 10, "Distance": 30},
                   "Card": {"Clusters": 8, "Distance": 10}}

    # Crop card and strip:
    for currentShape in boundingRects:
        # Get coordinates:
        x = boundingRects[currentShape][0]
        y = boundingRects[currentShape][1]
        w = boundingRects[currentShape][2]
        h = boundingRects[currentShape][3]

        # Crop the shape:
        currentCrop = gainDivision[y:y + h, x:x + w]
        showImage("Current Crop: " + currentShape, currentCrop)

        # Get k-means parameters according to shape:
        k = shapeParams[currentShape]["Clusters"]
        colorData = {"Threshold": [[245, 245, 245], [90, 84, 174]], "High": [255, 255, 255], "Low": [0, 0, 0]}
        minDistance = shapeParams[currentShape]["Distance"]

        # Segment image by grouping colors into k colors, apply distance filter:
        segmentedImage = imageQuantization(currentCrop, k, runDistanceFilter=True, filterParams=colorData,
                                           minDistance=minDistance)

        showImage("K-means: " + currentShape, segmentedImage)

        # # Get Y channel:
        # bgrdash = currentCrop.astype("float") / 255.0
        # K = 1 - np.max(bgrdash, axis=2)
        # yChannel = (1 - bgrdash[..., 0] - K) / (1 - K)
        # yChannel = (yChannel * 255).astype("uint8")
        #
        # # showImage("Y channel: " + currentShape, yChannel)
        #
        # # Otsu:
        # _, binaryImage = cv2.threshold(yChannel, 40, 255, cv2.THRESH_BINARY)
        # # showImage("binaryImage (Y): " + currentShape, binaryImage)
        #
        # # Get S channel:
        # hsvImage = cv2.cvtColor(currentCrop, cv2.COLOR_BGR2HSV)
        # H, S, V = cv2.split(hsvImage)
        #
        # # showImage("H channel: " + currentShape, S)

        # Otsu:
        # _, binaryImage = cv2.threshold(H, 40, 255, cv2.THRESH_BINARY)
        # # showImage("binaryImage (H): " + currentShape, binaryImage)
