/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

/**
 * @file image_processing_demo.cpp
 * @brief Comprehensive image processing demonstration
 * 
 * This example demonstrates core image processing operations in Neurova,
 * including loading, filtering, edge detection, and display.
 */

#include <neurova/neurova.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string imagePath = (argc > 1) ? argv[1] : "sample.jpg";
    
    //==========================================================================
    // 1. Image Loading and Basic Information
    //==========================================================================
    
    std::cout << "=== Image Loading ===" << std::endl;
    
    // Load image in color
    nv::Mat img = nv::imread(imagePath, nv::IMREAD_COLOR);
    
    if (img.empty()) {
        std::cerr << "Error: Could not load image: " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Image loaded successfully!" << std::endl;
    std::cout << "  Size: " << img.cols << " x " << img.rows << std::endl;
    std::cout << "  Channels: " << img.channels() << std::endl;
    std::cout << "  Depth: " << img.depth() << std::endl;
    std::cout << "  Type: " << img.type() << std::endl;
    
    //==========================================================================
    // 2. Color Space Conversions
    //==========================================================================
    
    std::cout << "\n=== Color Space Conversions ===" << std::endl;
    
    // Convert to grayscale
    nv::Mat gray;
    nv::cvtColor(img, gray, nv::COLOR_BGR2GRAY);
    std::cout << "Converted to grayscale" << std::endl;
    
    // Convert to HSV
    nv::Mat hsv;
    nv::cvtColor(img, hsv, nv::COLOR_BGR2HSV);
    std::cout << "Converted to HSV" << std::endl;
    
    // Convert to LAB
    nv::Mat lab;
    nv::cvtColor(img, lab, nv::COLOR_BGR2LAB);
    std::cout << "Converted to LAB" << std::endl;
    
    // Split HSV channels
    std::vector<nv::Mat> hsvChannels;
    nv::split(hsv, hsvChannels);
    std::cout << "Split HSV into " << hsvChannels.size() << " channels" << std::endl;
    
    //==========================================================================
    // 3. Image Filtering
    //==========================================================================
    
    std::cout << "\n=== Image Filtering ===" << std::endl;
    
    // Gaussian blur
    nv::Mat gaussian;
    nv::GaussianBlur(img, gaussian, nv::Size(5, 5), 1.5);
    std::cout << "Applied Gaussian blur (5x5, sigma=1.5)" << std::endl;
    
    // Median blur (good for salt-and-pepper noise)
    nv::Mat median;
    nv::medianBlur(img, median, 5);
    std::cout << "Applied median blur (5x5)" << std::endl;
    
    // Bilateral filter (edge-preserving)
    nv::Mat bilateral;
    nv::bilateralFilter(img, bilateral, 9, 75, 75);
    std::cout << "Applied bilateral filter" << std::endl;
    
    // Box blur
    nv::Mat boxBlur;
    nv::blur(img, boxBlur, nv::Size(5, 5));
    std::cout << "Applied box blur (5x5)" << std::endl;
    
    //==========================================================================
    // 4. Edge Detection
    //==========================================================================
    
    std::cout << "\n=== Edge Detection ===" << std::endl;
    
    // Canny edge detection
    nv::Mat edges;
    nv::Canny(gray, edges, 50, 150);
    std::cout << "Applied Canny edge detection" << std::endl;
    
    // Sobel gradients
    nv::Mat sobelX, sobelY, sobelMag;
    nv::Sobel(gray, sobelX, NV_16S, 1, 0, 3);
    nv::Sobel(gray, sobelY, NV_16S, 0, 1, 3);
    
    // Compute magnitude
    nv::Mat absX, absY;
    nv::convertScaleAbs(sobelX, absX);
    nv::convertScaleAbs(sobelY, absY);
    nv::addWeighted(absX, 0.5, absY, 0.5, 0, sobelMag);
    std::cout << "Applied Sobel gradient detection" << std::endl;
    
    // Laplacian
    nv::Mat laplacian;
    nv::Laplacian(gray, laplacian, NV_16S, 3);
    nv::convertScaleAbs(laplacian, laplacian);
    std::cout << "Applied Laplacian" << std::endl;
    
    //==========================================================================
    // 5. Morphological Operations
    //==========================================================================
    
    std::cout << "\n=== Morphological Operations ===" << std::endl;
    
    // Create structuring element
    nv::Mat kernel = nv::getStructuringElement(
        nv::MORPH_RECT, nv::Size(5, 5)
    );
    
    // Erosion
    nv::Mat eroded;
    nv::erode(edges, eroded, kernel, nv::Point(-1, -1), 1);
    std::cout << "Applied erosion" << std::endl;
    
    // Dilation
    nv::Mat dilated;
    nv::dilate(edges, dilated, kernel, nv::Point(-1, -1), 1);
    std::cout << "Applied dilation" << std::endl;
    
    // Opening (erosion followed by dilation)
    nv::Mat opened;
    nv::morphologyEx(edges, opened, nv::MORPH_OPEN, kernel);
    std::cout << "Applied opening" << std::endl;
    
    // Closing (dilation followed by erosion)
    nv::Mat closed;
    nv::morphologyEx(edges, closed, nv::MORPH_CLOSE, kernel);
    std::cout << "Applied closing" << std::endl;
    
    // Gradient (dilation - erosion)
    nv::Mat gradient;
    nv::morphologyEx(gray, gradient, nv::MORPH_GRADIENT, kernel);
    std::cout << "Applied morphological gradient" << std::endl;
    
    //==========================================================================
    // 6. Thresholding
    //==========================================================================
    
    std::cout << "\n=== Thresholding ===" << std::endl;
    
    // Simple binary threshold
    nv::Mat binary;
    nv::threshold(gray, binary, 127, 255, nv::THRESH_BINARY);
    std::cout << "Applied binary threshold at 127" << std::endl;
    
    // Otsu's automatic thresholding
    nv::Mat otsu;
    double otsuThresh = nv::threshold(gray, otsu, 0, 255, 
                                       nv::THRESH_BINARY | nv::THRESH_OTSU);
    std::cout << "Applied Otsu threshold (value: " << otsuThresh << ")" << std::endl;
    
    // Adaptive thresholding
    nv::Mat adaptive;
    nv::adaptiveThreshold(gray, adaptive, 255, 
                          nv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          nv::THRESH_BINARY, 11, 2);
    std::cout << "Applied adaptive threshold" << std::endl;
    
    //==========================================================================
    // 7. Geometric Transformations
    //==========================================================================
    
    std::cout << "\n=== Geometric Transformations ===" << std::endl;
    
    // Resize
    nv::Mat resized;
    nv::resize(img, resized, nv::Size(320, 240));
    std::cout << "Resized to 320x240" << std::endl;
    
    // Scale by factor
    nv::Mat scaled;
    nv::resize(img, scaled, nv::Size(), 0.5, 0.5, nv::INTER_LINEAR);
    std::cout << "Scaled by 0.5x" << std::endl;
    
    // Rotation
    nv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    nv::Mat rotMat = nv::getRotationMatrix2D(center, 45, 1.0);
    nv::Mat rotated;
    nv::warpAffine(img, rotated, rotMat, img.size());
    std::cout << "Rotated 45 degrees" << std::endl;
    
    // Flip
    nv::Mat flippedH, flippedV;
    nv::flip(img, flippedH, 1);  // Horizontal
    nv::flip(img, flippedV, 0);  // Vertical
    std::cout << "Applied horizontal and vertical flip" << std::endl;
    
    //==========================================================================
    // 8. Histogram Operations
    //==========================================================================
    
    std::cout << "\n=== Histogram Operations ===" << std::endl;
    
    // Compute histogram
    nv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange[] = {range};
    nv::calcHist(&gray, 1, nullptr, nv::Mat(), hist, 1, &histSize, histRange);
    std::cout << "Computed grayscale histogram" << std::endl;
    
    // Histogram equalization
    nv::Mat equalized;
    nv::equalizeHist(gray, equalized);
    std::cout << "Applied histogram equalization" << std::endl;
    
    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    nv::Ptr<nv::CLAHE> clahe = nv::createCLAHE(2.0, nv::Size(8, 8));
    nv::Mat claheResult;
    clahe->apply(gray, claheResult);
    std::cout << "Applied CLAHE" << std::endl;
    
    //==========================================================================
    // 9. Contour Detection
    //==========================================================================
    
    std::cout << "\n=== Contour Detection ===" << std::endl;
    
    // Find contours
    std::vector<std::vector<nv::Point>> contours;
    std::vector<nv::Vec4i> hierarchy;
    nv::findContours(edges.clone(), contours, hierarchy, 
                     nv::RETR_TREE, nv::CHAIN_APPROX_SIMPLE);
    std::cout << "Found " << contours.size() << " contours" << std::endl;
    
    // Draw contours
    nv::Mat contourImg = img.clone();
    nv::drawContours(contourImg, contours, -1, nv::Scalar(0, 255, 0), 2);
    
    // Analyze contours
    for (size_t i = 0; i < std::min(contours.size(), size_t(5)); i++) {
        double area = nv::contourArea(contours[i]);
        double perimeter = nv::arcLength(contours[i], true);
        nv::Rect boundingBox = nv::boundingRect(contours[i]);
        
        std::cout << "  Contour " << i << ": area=" << area 
                  << ", perimeter=" << perimeter 
                  << ", bbox=" << boundingBox << std::endl;
    }
    
    //==========================================================================
    // 10. Drawing Operations
    //==========================================================================
    
    std::cout << "\n=== Drawing Operations ===" << std::endl;
    
    nv::Mat drawing = img.clone();
    
    // Draw line
    nv::line(drawing, nv::Point(0, 0), nv::Point(100, 100), 
             nv::Scalar(0, 0, 255), 2);
    
    // Draw rectangle
    nv::rectangle(drawing, nv::Point(50, 50), nv::Point(200, 200),
                  nv::Scalar(0, 255, 0), 2);
    
    // Draw circle
    nv::circle(drawing, nv::Point(300, 200), 50, 
               nv::Scalar(255, 0, 0), -1);  // Filled
    
    // Draw ellipse
    nv::ellipse(drawing, nv::Point(400, 200), nv::Size(60, 30),
                45, 0, 360, nv::Scalar(255, 255, 0), 2);
    
    // Put text
    nv::putText(drawing, "Neurova Demo", nv::Point(50, 50),
                nv::FONT_HERSHEY_SIMPLEX, 1.5, nv::Scalar(255, 255, 255), 2);
    
    std::cout << "Added drawing elements" << std::endl;
    
    //==========================================================================
    // 11. Display Results
    //==========================================================================
    
    std::cout << "\n=== Displaying Results ===" << std::endl;
    
    // Create windows
    nv::namedWindow("Original", nv::WINDOW_AUTOSIZE);
    nv::namedWindow("Grayscale", nv::WINDOW_AUTOSIZE);
    nv::namedWindow("Edges", nv::WINDOW_AUTOSIZE);
    nv::namedWindow("Contours", nv::WINDOW_AUTOSIZE);
    nv::namedWindow("Drawing", nv::WINDOW_AUTOSIZE);
    
    // Display images
    nv::imshow("Original", img);
    nv::imshow("Grayscale", gray);
    nv::imshow("Edges", edges);
    nv::imshow("Contours", contourImg);
    nv::imshow("Drawing", drawing);
    
    std::cout << "Press any key to exit..." << std::endl;
    nv::waitKey(0);
    
    //==========================================================================
    // 12. Save Results
    //==========================================================================
    
    std::cout << "\n=== Saving Results ===" << std::endl;
    
    nv::imwrite("output_edges.png", edges);
    nv::imwrite("output_contours.png", contourImg);
    nv::imwrite("output_drawing.png", drawing);
    
    std::cout << "Results saved!" << std::endl;
    
    // Cleanup
    nv::destroyAllWindows();
    
    return 0;
}
