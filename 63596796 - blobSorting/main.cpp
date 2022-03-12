// File        :   main.cpp (Blob sorting)
// Version     :   1.0.0
// Description :   C++ program that extracts and sorts blob from left to right, trop to bottom
//                 Answer for: https://stackoverflow.com/q/63596796/12728244
// Date:       :   Mar 11, 2022
// Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
// License     :   Creative Commons CC0

#include <iostream>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Defines a resizeble image window:
void showImage( std::string imageTitle, cv::Mat inputImage ){
    cv::namedWindow(imageTitle, cv::WINDOW_NORMAL);
    cv::imshow(imageTitle, inputImage);
    cv::waitKey(0);
}

// Function to get the largest blob
cv::Mat findBiggestBlob( cv::Mat &inputImage ){

    // Deep copy of input:
    cv::Mat biggestBlob = inputImage.clone();

    int largestArea = 0;
    int largestContourIndex = 0;

    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    // Find the contours in the image
    cv::findContours( biggestBlob, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

    // Loop through the contours:
    for( int i = 0; i< (int)contours.size(); i++ ) {
        // Get contour area:
        double a = cv::contourArea( contours[i],false);
        // Keep the biggest contour/blob:
        if( a > largestArea ){
            largestArea = a;
            largestContourIndex = i;
        }
    }

    // Once you got the biggest blob, paint it black:
    cv::Mat tempMat = biggestBlob.clone();
    cv::drawContours( tempMat, contours, largestContourIndex, cv::Scalar(0),
                      CV_FILLED, 8, hierarchy );

    // Erase the smaller blobs:
    biggestBlob = biggestBlob - tempMat;

    tempMat.release();

    // Return the result:
    return biggestBlob;
}

int main()
{
    // Read the input image:
    std::string imageName = "D://opencvImages//E4kVKzN.png";
    cv::Mat testImage = cv::imread( imageName );

    // Compute grayscale image:
    cv::Mat grayImage;
    cv::cvtColor( testImage, grayImage, cv::COLOR_RGB2GRAY );

    // Get binary image via Otsu:
    cv::Mat binImage;
    cv::threshold( grayImage, binImage, 0, 255, cv::THRESH_OTSU );

    // Invert image:
    binImage = 255 - binImage;

    // Create a deep copy of the binary mask:
    cv::Mat rowMask = binImage.clone();

    // Morpholofy - horizontal dilation + erosion:
    int horizontalSize = 100; // a very big horizontal structuring element
    cv::Mat SE = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(horizontalSize,1) );
    cv::morphologyEx( rowMask, rowMask, cv::MORPH_DILATE, SE, cv::Point(-1,-1), 2 );
    cv::morphologyEx( rowMask, rowMask, cv::MORPH_ERODE, SE, cv::Point(-1,-1), 1 );

    // Label the row mask:
    int rowCount = 0; //This will count our rows

    // Loop through the mask:
    for( int y = 0; y < rowMask.rows; y++ ){
        for( int x = 0; x < rowMask.cols; x++ ){
            // Get the current pixel:
            uchar currentPixel = rowMask.at<uchar>( y, x );
            // If the pixel is white, this is an unlabeled blob:
            if ( currentPixel == 255 ) {
                // Create new label (different from zero):
                rowCount++;
                // Flood fill on this point:
                cv::floodFill( rowMask, cv::Point( x, y ), rowCount, (cv::Rect*)0, cv::Scalar(), 0 );
            }
        }
    }

    // Create rows image:
    cv::Mat rowTable = cv::Mat::zeros( cv::Size(binImage.cols, rowCount), CV_8UC1 );
    // Just for convenience:
    rowTable = 255 - rowTable;

    // Prepare a couple of dictionaries for data storing:
    std::map< int, cv::Point > blobMap; // holds label, gives centroid
    std::map< int, cv::Rect > boundingBoxMap; // holds label, gives bounding box

    // Extract each individual blob:
    cv::Mat bobFilterInput = binImage.clone();

    // The new blob label:
    int blobLabel = 0;

    // Some control variables:
    bool extractBlobs = true; //Controls loop
    int currentBlob = 0; //Counter of blobs

    // Run process while there are blobs in the image:
    while ( extractBlobs ){

        // Get the biggest blob:
        cv::Mat biggestBlob = findBiggestBlob( bobFilterInput );

        // Compute the centroid/center of mass:
        cv::Moments momentStructure = cv::moments( biggestBlob, true );
        float cx = momentStructure.m10 / momentStructure.m00;
        float cy = momentStructure.m01 / momentStructure.m00;

        // Centroid point:
        cv::Point blobCentroid;
        blobCentroid.x = cx;
        blobCentroid.y = cy;

        // Get the blob's bounding rectangle:
        cv::Rect cropBox = cv::boundingRect(biggestBlob);

        // Label blob:
        blobLabel++;
        blobMap.emplace( blobLabel, blobCentroid );
        boundingBoxMap.emplace( blobLabel, cropBox );

        // Get the row for this centroid
        int blobRow = rowMask.at<uchar>( cy, cx );
        blobRow--;

        // Place centroid on rowed image:
        rowTable.at<uchar>( blobRow, cx ) = blobLabel;

        // Resume blob flow control:
        cv::Mat blobDifference = bobFilterInput - biggestBlob;
        // How many pixels are left on the new mask?
        int pixelsLeft = cv::countNonZero( blobDifference );
        bobFilterInput = blobDifference;

        // Done extracting blobs?
        if ( pixelsLeft <= 0 ){
            extractBlobs = false;
        }

        // Increment blob counter:
        currentBlob++;

    }

    // Got order of blobs, now numerate them:
    int blobCounter = 1; //The ORDERED label, starting at 1
    for( int y = 0; y < rowTable.rows; y++ ){
        for( int x = 0; x < rowTable.cols; x++ ){

            //Get current label:
            uchar currentLabel = rowTable.at<uchar>( y, x );
            //Is it a valid label?

            if ( currentLabel != 255 ){
                //Get the bounding box for this label:
                cv::Rect currentBoundingBox = boundingBoxMap[ currentLabel ];
                cv::rectangle( testImage, currentBoundingBox, cv::Scalar(0,255,0), 2, 8, 0 );
                //The blob counter to string:
                std::string counterString = std::to_string( blobCounter );
                cv::putText( testImage, counterString, cv::Point( currentBoundingBox.x, currentBoundingBox.y-1 ),
                             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 1, cv::LINE_8, false );
                blobCounter++; //Increment the blob/label
                //Show image:
                showImage("Ordered Blobs", testImage);
            }

        }

    }


}
