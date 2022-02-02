// File        :   main.cpp (Rubik's Cube Segmentation)
// Version     :   1.0.0
// Description :   C++ program that detects and segments the 9 cells of a Rubik's cube.
//                 Answer for: https://stackoverflow.com/questions/64564138/how-do-i-split-up-thresholds-into-squares-in-opencv2/64567585
// Date:       :   Jan 26, 2022
// Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
// License     :   Creative Commons CC0

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Defines a resizeble image window:
void showImage( std::string imageTitle, cv::Mat inputImage ){
    cv::namedWindow(imageTitle, cv::WINDOW_NORMAL);
    cv::imshow(imageTitle, inputImage);
    cv::waitKey(0);
}

int main()
{
    // Read the input image:
    std::string imageName = "D://opencvImages//cube.jpg";
    cv::Mat testImage =  cv::imread( imageName );

    // Convert BGR to Gray:
    cv::Mat grayImage;
    cv::cvtColor( testImage, grayImage, cv::COLOR_RGB2GRAY );

    // Apply Gaussian blur with a X-Y Sigma of 50:
    cv::GaussianBlur( grayImage, grayImage, cv::Size(3,3), 50, 50 );

    // Prepare edges matrix:
    cv::Mat testEdges;

    // Setup lower and upper thresholds for edge detection:
    float lowerThreshold = 20;
    float upperThreshold = 3 * lowerThreshold;

    // Get Edges via Canny:
    cv::Canny( grayImage, testEdges, lowerThreshold, upperThreshold );

    // Prepare a rectangular, 3x3 structuring element:
    cv::Mat SE = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3, 3) );

    // OP iterations:
    int dilateIterations = 5;

    // Prepare the dilation matrix:
    cv::Mat binDilation;

    // Perform the morph operation:
    cv::morphologyEx( testEdges, binDilation, cv::MORPH_DILATE, SE, cv::Point(-1,-1), dilateIterations );
    // Show image:
    showImage("Dilation 1", binDilation);

    std::vector< std::vector<cv::Point> > imageCorners;
    imageCorners.push_back( { cv::Point(0,0), cv::Point(binDilation.cols,0) } );
    imageCorners.push_back( { cv::Point(binDilation.cols,0), cv::Point(binDilation.cols, binDilation.rows) } );
    imageCorners.push_back( { cv::Point(binDilation.cols, binDilation.rows), cv::Point(0,binDilation.rows) } );
    imageCorners.push_back( { cv::Point(0,binDilation.rows), cv::Point(0, 0) } );

    // Define the SUPER THICKNESS:
    int lineThicness  = 200;

    // Loop through my line coordinates and draw four lines at the borders:
    for ( int c = 0 ; c < 4 ; c++ ){
        // Get current vector of points:
        std::vector<cv::Point> currentVect = imageCorners[c];
        // Get the starting/ending points:
        cv::Point startPoint = currentVect[0];
        cv::Point endPoint = currentVect[1];
        // Draw the line:
        cv::line( binDilation, startPoint, endPoint, cv::Scalar(255,255,255), lineThicness );
    }

    // Show image:
    showImage("Dilation 2", binDilation);

    // Set the offset of the image corners. Ensure the area to be filled is black:
    int fillOffsetX = 200;
    int fillOffsetY = 200;
    cv::Scalar fillTolerance = 0; //No tolerance
    int fillColor = 255; //Fill color is white

    // Get the dimensions of the image:
    int targetCols = binDilation.cols;
    int targetRows = binDilation.rows;

    // Flood-fill at the four corners of the image:
    cv::floodFill( binDilation, cv::Point( fillOffsetX, fillOffsetY ), fillColor, (cv::Rect*)0, fillTolerance, fillTolerance);
    cv::floodFill( binDilation, cv::Point( fillOffsetX, targetRows - fillOffsetY ), fillColor, (cv::Rect*)0, fillTolerance, fillTolerance);
    cv::floodFill( binDilation, cv::Point( targetCols - fillOffsetX, fillOffsetY ), fillColor, (cv::Rect*)0, fillTolerance, fillTolerance);
    cv::floodFill( binDilation, cv::Point( targetCols - fillOffsetX, targetRows - fillOffsetY ), fillColor, (cv::Rect*)0, fillTolerance, fillTolerance);

    // Show image:
    showImage("Filled Mask", binDilation);

    // Get the inverted image:
    cv::Mat cubeMask = 255 - binDilation;

    // Set some really high iterations here:
    int closeIterations = 50;

    // Dilate
    cv::morphologyEx( cubeMask, cubeMask, cv::MORPH_DILATE, SE, cv::Point(-1,-1), closeIterations );
    // Erode:
    cv::morphologyEx( cubeMask, cubeMask, cv::MORPH_ERODE, SE, cv::Point(-1,-1), closeIterations );

    // Show image:
    showImage("Filtered Mask", cubeMask);

    // Lets get the blob contour:
    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours( cubeMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    // There should be only one contour, the item number 0:
    cv::Rect boundigRect = cv::boundingRect( contours[0] );

    // Number of squares or "grids"
    int verticalGrids = 3;
    int horizontalGrids = 3;

    // Grid dimensions:
    float gridWidth = (float)boundigRect.width / 3.0;
    float gridHeight = (float)boundigRect.height / 3.0;

    // Grid counter:
    int gridCounter = 1;

    // Loop thru vertical dimension:
    for ( int j = 0; j < verticalGrids; ++j ) {

        // Grid starting Y:
        int yo = j * gridHeight;

        // Loop thru horizontal dimension:
        for ( int i = 0; i < horizontalGrids; ++i ) {

            // Grid starting X:
            int xo = i * gridWidth;

            // Grid dimensions:
            cv::Rect gridBox;
            gridBox.x = boundigRect.x + xo;
            gridBox.y = boundigRect.y + yo;
            gridBox.width = gridWidth;
            gridBox.height = gridHeight;

            // Draw a rectangle using the grid dimensions:
            cv::rectangle( testImage, gridBox, cv::Scalar(0,0,255), 5 );

            // Int to string:
            std::string gridCounterString = std::to_string( gridCounter );

            // String position:
            cv::Point textPosition;
            textPosition.x = gridBox.x + 0.5 * gridBox.width;
            textPosition.y = gridBox.y + 0.5 * gridBox.height;

            // Draw string:
            cv::putText( testImage, gridCounterString, textPosition, cv::FONT_HERSHEY_SIMPLEX,
                         1, cv::Scalar(255,0,0), 3, cv::LINE_8, false );

            gridCounter++;

            // Show the grid on the image:
            showImage( "Cube Grid", testImage );

        }

    }
    
    return 0;

}
