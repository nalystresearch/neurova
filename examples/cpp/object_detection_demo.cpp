/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * @file object_detection_demo.cpp
 * @brief Object detection using detectionv8 with Neurova DNN module
 * 
 * This example demonstrates how to:
 * - Load a detectionv8 ONNX model
 * - Configure GPU backend
 * - Process images and video
 * - Apply NMS and draw detections
 */

#include <neurova/neurova.hpp>
#include <fstream>
#include <iostream>
#include <vector>

// COCO class names
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

// Generate random colors for each class
std::vector<nv::Scalar> generateColors(int numClasses) {
    std::vector<nv::Scalar> colors;
    std::srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < numClasses; i++) {
        colors.push_back(nv::Scalar(
            std::rand() % 256,
            std::rand() % 256,
            std::rand() % 256
        ));
    }
    return colors;
}

// Structure to hold detection results
struct Detection {
    int classId;
    float confidence;
    nv::Rect box;
};

// Post-process detectionv8 output
std::vector<Detection> postProcess(
    const nv::Mat& output, 
    float confThreshold, 
    float nmsThreshold,
    int imgWidth, 
    int imgHeight,
    int inputWidth,
    int inputHeight
) {
    std::vector<Detection> detections;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<nv::Rect> boxes;
    
    // detectionv8 output shape: [1, 84, 8400] where 84 = 4 (box) + 80 (classes)
    // Transpose to [8400, 84]
    int numDetections = output.cols;
    int numClasses = output.rows - 4;
    
    for (int i = 0; i < numDetections; i++) {
        // Get class scores for this detection
        float maxScore = 0;
        int maxClassId = -1;
        
        for (int j = 4; j < output.rows; j++) {
            float score = output.at<float>(j, i);
            if (score > maxScore) {
                maxScore = score;
                maxClassId = j - 4;
            }
        }
        
        if (maxScore >= confThreshold) {
            // Get bounding box
            float cx = output.at<float>(0, i);
            float cy = output.at<float>(1, i);
            float w = output.at<float>(2, i);
            float h = output.at<float>(3, i);
            
            // Convert to image coordinates
            float scaleX = static_cast<float>(imgWidth) / inputWidth;
            float scaleY = static_cast<float>(imgHeight) / inputHeight;
            
            int x = static_cast<int>((cx - w / 2) * scaleX);
            int y = static_cast<int>((cy - h / 2) * scaleY);
            int width = static_cast<int>(w * scaleX);
            int height = static_cast<int>(h * scaleY);
            
            // Clamp to image bounds
            x = std::max(0, x);
            y = std::max(0, y);
            width = std::min(width, imgWidth - x);
            height = std::min(height, imgHeight - y);
            
            boxes.push_back(nv::Rect(x, y, width, height));
            confidences.push_back(maxScore);
            classIds.push_back(maxClassId);
        }
    }
    
    // Apply NMS
    std::vector<int> indices;
    nv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    for (int idx : indices) {
        Detection det;
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        detections.push_back(det);
    }
    
    return detections;
}

// Draw detections on image
void drawDetections(
    nv::Mat& image, 
    const std::vector<Detection>& detections,
    const std::vector<nv::Scalar>& colors
) {
    for (const auto& det : detections) {
        // Get color for this class
        nv::Scalar color = colors[det.classId % colors.size()];
        
        // Draw bounding box
        nv::rectangle(image, det.box, color, 2);
        
        // Create label
        std::string label = CLASS_NAMES[det.classId] + ": " + 
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        
        // Get text size
        int baseline;
        nv::Size textSize = nv::getTextSize(label, nv::FONT_HERSHEY_SIMPLEX, 
                                            0.5, 1, &baseline);
        
        // Draw label background
        int labelY = std::max(det.box.y, textSize.height + 10);
        nv::rectangle(image, 
                      nv::Point(det.box.x, labelY - textSize.height - 10),
                      nv::Point(det.box.x + textSize.width, labelY),
                      color, -1);
        
        // Draw label text
        nv::putText(image, label, nv::Point(det.box.x, labelY - 5),
                    nv::FONT_HERSHEY_SIMPLEX, 0.5, nv::Scalar(255, 255, 255), 1);
    }
}

// Print available backends and targets
void printAvailableBackends() {
    std::cout << "Available DNN backends:" << std::endl;
    auto backends = nv::dnn::getAvailableBackends();
    for (const auto& [backend, target] : backends) {
        std::string backendName, targetName;
        
        switch (backend) {
            case nv::dnn::DNN_BACKEND_DEFAULT: backendName = "Default"; break;
            case nv::dnn::DNN_BACKEND_DEFAULT: backendName = "Default"; break;
            case nv::dnn::DNN_BACKEND_CUDA: backendName = "CUDA"; break;
            case nv::dnn::DNN_BACKEND_VKCOM: backendName = "Vulkan"; break;
            default: backendName = "Unknown"; break;
        }
        
        switch (target) {
            case nv::dnn::DNN_TARGET_CPU: targetName = "CPU"; break;
            case nv::dnn::DNN_TARGET_OPENCL: targetName = "OpenCL"; break;
            case nv::dnn::DNN_TARGET_CUDA: targetName = "CUDA"; break;
            case nv::dnn::DNN_TARGET_VULKAN: targetName = "Vulkan"; break;
            default: targetName = "Unknown"; break;
        }
        
        std::cout << "  " << backendName << " -> " << targetName << std::endl;
    }
}

int main(int argc, char** argv) {
    // Configuration
    const std::string modelPath = (argc > 1) ? argv[1] : "detectionv8n.onnx";
    const std::string inputPath = (argc > 2) ? argv[2] : "input.jpg";
    const float confThreshold = 0.5f;
    const float nmsThreshold = 0.45f;
    const int inputWidth = 640;
    const int inputHeight = 640;
    
    std::cout << "=== Neurova Object Detection Demo ===" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Input: " << inputPath << std::endl;
    std::cout << std::endl;
    
    // Print available backends
    printAvailableBackends();
    std::cout << std::endl;
    
    // Generate colors for drawing
    auto colors = generateColors(CLASS_NAMES.size());
    
    //==========================================================================
    // Load Model
    //==========================================================================
    
    std::cout << "Loading model..." << std::endl;
    
    nv::dnn::Net net;
    try {
        net = nv::dnn::readNet(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }
    
    // Configure backend (try CUDA first, fall back to CPU)
    bool useCuda = false;
#ifdef NV_CUDA_ENABLED
    try {
        net.setPreferableBackend(nv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(nv::dnn::DNN_TARGET_CUDA);
        useCuda = true;
        std::cout << "Using CUDA backend" << std::endl;
    } catch (...) {
        std::cout << "CUDA not available, falling back to CPU" << std::endl;
    }
#endif
    
    if (!useCuda) {
        net.setPreferableBackend(nv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(nv::dnn::DNN_TARGET_CPU);
        std::cout << "Using CPU backend" << std::endl;
    }
    
    // Print network info
    auto layerNames = net.getLayerNames();
    std::cout << "Network has " << layerNames.size() << " layers" << std::endl;
    
    auto outputNames = net.getUnconnectedOutLayersNames();
    std::cout << "Output layers: ";
    for (const auto& name : outputNames) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    //==========================================================================
    // Determine Input Type (Image or Video)
    //==========================================================================
    
    bool isVideo = (inputPath.find(".mp4") != std::string::npos ||
                    inputPath.find(".avi") != std::string::npos ||
                    inputPath.find(".mov") != std::string::npos ||
                    inputPath == "0");  // Camera
    
    if (isVideo || inputPath == "0") {
        //======================================================================
        // Video/Camera Processing
        //======================================================================
        
        std::cout << "=== Video Processing Mode ===" << std::endl;
        
        nv::VideoCapture cap;
        if (inputPath == "0") {
            cap.open(0);  // Camera
        } else {
            cap.open(inputPath);
        }
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video source" << std::endl;
            return -1;
        }
        
        double fps = cap.get(nv::CAP_PROP_FPS);
        int frameWidth = static_cast<int>(cap.get(nv::CAP_PROP_FRAME_WIDTH));
        int frameHeight = static_cast<int>(cap.get(nv::CAP_PROP_FRAME_HEIGHT));
        
        std::cout << "Video: " << frameWidth << "x" << frameHeight 
                  << " @ " << fps << " FPS" << std::endl;
        
        nv::Mat frame;
        int frameCount = 0;
        double totalTime = 0;
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Create blob
            nv::Mat blob = nv::dnn::blobFromImage(
                frame, 1.0/255.0, nv::Size(inputWidth, inputHeight),
                nv::Scalar(), true, false
            );
            
            // Run inference
            net.setInput(blob);
            std::vector<nv::Mat> outputs;
            net.forward(outputs, outputNames);
            
            // Post-process
            auto detections = postProcess(
                outputs[0], confThreshold, nmsThreshold,
                frame.cols, frame.rows, inputWidth, inputHeight
            );
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double inferenceTime = std::chrono::duration<double, std::milli>(
                endTime - startTime).count();
            
            totalTime += inferenceTime;
            frameCount++;
            
            // Draw detections
            drawDetections(frame, detections, colors);
            
            // Draw FPS
            double avgFps = 1000.0 * frameCount / totalTime;
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(avgFps));
            nv::putText(frame, fpsText, nv::Point(10, 30),
                        nv::FONT_HERSHEY_SIMPLEX, 1.0, nv::Scalar(0, 255, 0), 2);
            
            // Display
            nv::imshow("Object Detection", frame);
            
            if (nv::waitKey(1) == 'q') break;
        }
        
        cap.release();
        
        std::cout << "\nProcessed " << frameCount << " frames" << std::endl;
        std::cout << "Average FPS: " << (1000.0 * frameCount / totalTime) << std::endl;
        
    } else {
        //======================================================================
        // Single Image Processing
        //======================================================================
        
        std::cout << "=== Image Processing Mode ===" << std::endl;
        
        nv::Mat image = nv::imread(inputPath);
        if (image.empty()) {
            std::cerr << "Error: Could not load image: " << inputPath << std::endl;
            return -1;
        }
        
        std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
        
        // Warmup run
        nv::Mat warmupBlob = nv::dnn::blobFromImage(
            image, 1.0/255.0, nv::Size(inputWidth, inputHeight),
            nv::Scalar(), true, false
        );
        net.setInput(warmupBlob);
        net.forward();
        std::cout << "Warmup complete" << std::endl;
        
        // Timed inference
        auto startTime = std::chrono::high_resolution_clock::now();
        
        nv::Mat blob = nv::dnn::blobFromImage(
            image, 1.0/255.0, nv::Size(inputWidth, inputHeight),
            nv::Scalar(), true, false
        );
        
        net.setInput(blob);
        std::vector<nv::Mat> outputs;
        net.forward(outputs, outputNames);
        
        auto detections = postProcess(
            outputs[0], confThreshold, nmsThreshold,
            image.cols, image.rows, inputWidth, inputHeight
        );
        
        auto endTime = std::chrono::high_resolution_clock::now();
        double inferenceTime = std::chrono::duration<double, std::milli>(
            endTime - startTime).count();
        
        std::cout << "\nInference time: " << inferenceTime << " ms" << std::endl;
        std::cout << "Detected " << detections.size() << " objects" << std::endl;
        
        // Print detections
        for (const auto& det : detections) {
            std::cout << "  " << CLASS_NAMES[det.classId] 
                      << " (" << static_cast<int>(det.confidence * 100) << "%)"
                      << " at [" << det.box.x << ", " << det.box.y 
                      << ", " << det.box.width << ", " << det.box.height << "]"
                      << std::endl;
        }
        
        // Draw and display
        drawDetections(image, detections, colors);
        
        nv::imshow("Object Detection", image);
        nv::imwrite("output_detection.jpg", image);
        std::cout << "\nResult saved to output_detection.jpg" << std::endl;
        
        std::cout << "Press any key to exit..." << std::endl;
        nv::waitKey(0);
    }
    
    nv::destroyAllWindows();
    
    return 0;
}
