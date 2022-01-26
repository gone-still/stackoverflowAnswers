# imports:
import numpy as np
import cv2

# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)

# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)

# image path
path = "D://opencvImages//"
fileName = "mx8lW.jpg"

# Reading an image in default mode:
inputImage = cv2.imread(path + fileName)
inputImageCopy = inputImage.copy()

# Convert RGB to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

showImage("grayscaleImage", grayscaleImage)

# Crop ROI
(imageHeight, imageWidth) = grayscaleImage.shape[:2]
roiX = 0
roiY = int(0.05 * imageHeight)
roiWidth = imageWidth
roiHeight = int(0.05 * imageHeight)

color = (0, 255, 0)
cv2.rectangle(inputImageCopy, (int(roiX), int(roiY)), (int(roiX + roiWidth), int(roiY + roiHeight)), color, 5)
showImage("inputImageCopy", inputImageCopy)

writeImage(path+"pageRectangle", inputImageCopy)

imageRoi = grayscaleImage[roiY:roiY+roiHeight, roiX:roiWidth]

showImage("ImageROI", imageRoi)

writeImage(path+"pageRoi", imageRoi)

# Thresholding:
_, binaryImage = cv2.threshold(imageRoi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

showImage("binaryImage", binaryImage)
writeImage(path+"binaryPage", binaryImage)

# Reduce the ROI to a n row x 1 columns matrix:
reducedImg = cv2.reduce(binaryImage, 0, cv2.REDUCE_MAX)

showImage("reducedImg", reducedImg)
writeImage(path+"pageReduced", reducedImg)

# Store the transition positions here:
linePositions = []

# Find transitions from 0 to 255:
pastPixel = 255
for x in range(reducedImg.shape[1]):
    # Get current pixel:
    currentPixel = reducedImg[0,x]
    # Check for the "jumps":
    if currentPixel == 255 and pastPixel == 0:
        # Store the jump locations in list:
        print("Got Jump at:"+str(x))
        linePositions.append(x)
    # Set current pixel to past pixel:
    pastPixel = currentPixel


# Crop pages:
for i in range(len(linePositions)):
    # Get top left:
    cropX = linePositions[i]

    # Get top left:
    if i != len(linePositions)-1:
        cropWidth = linePositions[i+1]
    else:
        cropWidth = reducedImg.shape[1]

    # Crop page:
    cropY = 0
    cropHeight = imageHeight
    currentCrop = inputImage[cropY:cropHeight,cropX:cropWidth]

    showImage("CurrentCrop", currentCrop)
    writeImage(path+"pagecurrentCrop"+str(i), currentCrop)