// File        :   main.cpp (Sepia Filter)
// Version     :   1.0.0
// Description :   C++ program that implements a sepia filter
//                 Answer for: https://stackoverflow.com/q/77836672/
// Date:       :   Jan 30, 2024
// Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
// License     :   Creative Commons CC0

#include <iostream>
#include <opencv2/opencv.hpp>

// Defines a resizeble image window:
void showImage( std::string imageTitle, cv::Mat inputImage ){
    cv::namedWindow(imageTitle, cv::WINDOW_NORMAL);
    cv::imshow(imageTitle, inputImage);
    cv::waitKey(0);
}


int main()
{

    // Read image:
    std::string imagePath = "D://opencvImages//swedishsnoop.png";
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);

    // Create a new Mat to store the sepia-toned image
    cv::Mat sepiaToned = inputImage.clone();

    // Iterate through each pixel
    for (int y = 0; y < inputImage.rows; y++)
    {
        for (int x = 0; x < inputImage.cols; x++)
        {
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);

            // Apply sepia tone filter
            float blue = pixel[0];
            float green = pixel[1];
            float red = pixel[2];

            float newR = (0.393 * red) + (0.769 * green) + (0.189 * blue);
            float newG = (0.349 * red) + (0.686 * green) + (0.168 * blue);
            float newB = (0.272 * red) + (0.534 * green) + (0.131 * blue);

            // Properly saturate cast pixel intensities:
            newB = cv::saturate_cast<uchar>(newB);
            newG = cv::saturate_cast<uchar>(newG);
            newR = cv::saturate_cast<uchar>(newR);

            // Update pixel values in the new Mat
            sepiaToned.at<cv::Vec3b>(y, x) = cv::Vec3b(newB, newG, newR);
        }
    }

    // Display the original and sepia-toned images
    showImage("Original Image", inputImage);
    showImage("Sepia-Toned Image", sepiaToned);

    return 0;

}