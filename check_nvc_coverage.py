#!/usr/bin/env python3
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""check neurova.nvc coverage against standard cv modules."""

import neurova.nvc as nvc
import neurova.dnn as dnn

# total exports
all_exports = [x for x in dir(nvc) if not x.startswith('_')]
print(f"total neurova.nvc exports: {len(all_exports)}")
print("=" * 60)

def check_module(name, functions):
    present = [f for f in functions if hasattr(nvc, f)]
    missing = [f for f in functions if not hasattr(nvc, f)]
    pct = len(present) / len(functions) * 100 if functions else 0
    print(f"\n{name}: {len(present)}/{len(functions)} ({pct:.0f}%)")
    if missing:
        print(f"  missing: {missing}")
    return present, missing

all_present = []
all_missing = []

# Core operations
p, m = check_module("core", [
    "add", "subtract", "multiply", "divide", "addWeighted", "absdiff", "convertScaleAbs",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", 
    "flip", "rotate", "split", "merge", "minMaxLoc", "normalize", "countNonZero",
    "mean", "meanStdDev", "copyMakeBorder", "LUT", "magnitude", "phase", 
    "cartToPolar", "polarToCart",
    # Additional critical
    "hconcat", "vconcat", "reduce", "repeat", "transpose", "gemm",
    "exp", "log", "pow", "sqrt", "norm", "compare", "sum", "min", "max",
    "findNonZero", "inRange", "setIdentity", "determinant", "trace", "solve",
    "eigen", "SVDecomp", "PCACompute", "dct", "dft", "idct", "idft",
])
all_present.extend(p)
all_missing.extend([("core", f) for f in m])

# Image Processing - Filtering
p, m = check_module("imgproc/filtering", [
    "GaussianBlur", "blur", "medianBlur", "bilateralFilter", "filter2D", "boxFilter",
    "Sobel", "Scharr", "Laplacian", "Canny", "getStructuringElement", 
    "getGaussianKernel", "getDerivKernels", "sepFilter2D", "getGaborKernel",
])
all_present.extend(p)
all_missing.extend([("imgproc/filtering", f) for f in m])

# Image Processing - Morphology
p, m = check_module("imgproc/morphology", [
    "erode", "dilate", "morphologyEx",
])
all_present.extend(p)
all_missing.extend([("imgproc/morphology", f) for f in m])

# Image Processing - Geometric
p, m = check_module("imgproc/geometric", [
    "resize", "warpAffine", "warpPerspective", "remap",
    "getAffineTransform", "getPerspectiveTransform", "getRotationMatrix2D",
    "invertAffineTransform", "warpPolar", "logPolar", "linearPolar",
])
all_present.extend(p)
all_missing.extend([("imgproc/geometric", f) for f in m])

# Image Processing - Drawing
p, m = check_module("imgproc/drawing", [
    "line", "arrowedLine", "rectangle", "circle", "ellipse", 
    "polylines", "fillPoly", "fillConvexPoly", "putText", "getTextSize",
    "drawMarker", "drawContours",
])
all_present.extend(p)
all_missing.extend([("imgproc/drawing", f) for f in m])

# Image Processing - Contours/Shape
p, m = check_module("imgproc/contours", [
    "findContours", "contourArea", "arcLength", "boundingRect",
    "minAreaRect", "minEnclosingCircle", "convexHull", "approxPolyDP",
    "moments", "isContourConvex", "pointPolygonTest", "fitLine",
    "convexityDefects", "matchShapes", "connectedComponents", "connectedComponentsWithStats",
])
all_present.extend(p)
all_missing.extend([("imgproc/contours", f) for f in m])

# Image Processing - Color/Threshold
p, m = check_module("imgproc/color_threshold", [
    "cvtColor", "threshold", "adaptiveThreshold", "inRange",
])
all_present.extend(p)
all_missing.extend([("imgproc/color_threshold", f) for f in m])

# Image Processing - Pyramids/Segmentation
p, m = check_module("imgproc/pyramids_segmentation", [
    "pyrDown", "pyrUp", "buildPyramid",
    "distanceTransform", "watershed", "grabCut", "floodFill", "pyrMeanShiftFiltering",
])
all_present.extend(p)
all_missing.extend([("imgproc/pyramids_segmentation", f) for f in m])

# Image Processing - Histograms
p, m = check_module("imgproc/histograms", [
    "calcHist", "compareHist", "equalizeHist", "calcBackProject", "createCLAHE",
])
all_present.extend(p)
all_missing.extend([("imgproc/histograms", f) for f in m])

# Image Processing - Feature Detection
p, m = check_module("imgproc/features", [
    "HoughLines", "HoughLinesP", "HoughCircles", "matchTemplate",
    "cornerHarris", "cornerMinEigenVal", "goodFeaturesToTrack", "cornerSubPix",
    "createLineSegmentDetector",
])
all_present.extend(p)
all_missing.extend([("imgproc/features", f) for f in m])

# Video - Motion/Tracking
p, m = check_module("video", [
    "calcOpticalFlowPyrLK", "calcOpticalFlowFarneback", "buildOpticalFlowPyramid",
    "createBackgroundSubtractorMOG2", "createBackgroundSubtractorKNN",
    "CamShift", "meanShift", "KalmanFilter",
    # Trackers (common implementations)
    "TrackerMIL", "TrackerKCF", "TrackerCSRT", "TrackerGOTURN",
])
all_present.extend(p)
all_missing.extend([("video", f) for f in m])

# Calib3d
p, m = check_module("calib3d", [
    "findChessboardCorners", "findChessboardCornersSB", "cornerSubPix", "drawChessboardCorners",
    "calibrateCamera", "getOptimalNewCameraMatrix", "undistort", "undistortPoints",
    "solvePnP", "solvePnPRansac", "projectPoints",
    "findHomography", "findFundamentalMat", "findEssentialMat",
    "Rodrigues", "decomposeHomographyMat", "triangulatePoints",
    "stereoCalibrate", "stereoRectify", "computeCorrespondEpilines",
    "estimateAffine2D", "estimateAffinePartial2D", "reprojectImageTo3D",
])
all_present.extend(p)
all_missing.extend([("calib3d", f) for f in m])

# Features2d
p, m = check_module("features2d", [
    "ORB_create", "SIFT_create", "AKAZE_create", "BRISK_create",
    "BFMatcher_create", "FlannBasedMatcher_create",
    "drawKeypoints", "drawMatches", "drawMatchesKnn",
    "FastFeatureDetector_create", "GFTTDetector_create", "SimpleBlobDetector_create",
])
all_present.extend(p)
all_missing.extend([("features2d", f) for f in m])

# Objdetect
p, m = check_module("objdetect", [
    "CascadeClassifier", "HOGDescriptor", "groupRectangles",
    "QRCodeDetector", "FaceDetectorYN", "FaceRecognizerSF",
])
all_present.extend(p)
all_missing.extend([("objdetect", f) for f in m])

# DNN (check in dnn module)
dnn_fns = ["Net", "readNet", "readNetFromCaffe", "readNetFromNeurova", 
           "readNetFromONNX", "readNetFromDarknet", "blobFromImage", "blobFromImages", 
           "NMSBoxes"]
dnn_present = [f for f in dnn_fns if hasattr(dnn, f)]
dnn_missing = [f for f in dnn_fns if not hasattr(dnn, f)]
pct = len(dnn_present) / len(dnn_fns) * 100
print(f"\ndnn: {len(dnn_present)}/{len(dnn_fns)} ({pct:.0f}%)")
if dnn_missing:
    print(f"  Missing: {dnn_missing}")
all_present.extend(dnn_present)
all_missing.extend([("dnn", f) for f in dnn_missing])

# Photo
p, m = check_module("photo", [
    "fastNlMeansDenoising", "fastNlMeansDenoisingColored",
    "inpaint", "seamlessClone", "colorChange", "illuminationChange",
    "edgePreservingFilter", "detailEnhance", "pencilSketch", "stylization",
    "createTonemap", "createTonemapDrago", "createTonemapReinhard",
    "createMergeDebevec", "createMergeMertens",
])
all_present.extend(p)
all_missing.extend([("photo", f) for f in m])

# Stitching
p, m = check_module("stitching", [
    "Stitcher", "createStitcher",
])
all_present.extend(p)
all_missing.extend([("stitching", f) for f in m])

# I/O & HighGUI
p, m = check_module("io/highgui", [
    "imread", "imwrite", "VideoCapture", "VideoWriter", "VideoWriter_fourcc",
    "imshow", "waitKey", "waitKeyEx", "namedWindow", "destroyWindow", "destroyAllWindows",
    "moveWindow", "resizeWindow", "createTrackbar", "getTrackbarPos", "setMouseCallback",
])
all_present.extend(p)
all_missing.extend([("io/highgui", f) for f in m])

# Summary
print("\n" + "=" * 60)
print(f"SUMMARY: {len(all_present)} present, {len(all_missing)} missing")
print(f"Overall Coverage: {len(all_present)/(len(all_present)+len(all_missing))*100:.1f}%")

print("\n" + "=" * 60)
print("MISSING CRITICAL FUNCTIONS BY PRIORITY:")
print("=" * 60)

critical = ["hconcat", "vconcat", "transpose", "reduce", "norm", "compare",
            "exp", "log", "sqrt", "pow", "sum", "min", "max",
            "sepFilter2D", "fitLine", "convexityDefects", "buildPyramid",
            "stereoCalibrate", "stereoRectify", "estimateAffine2D",
            "TrackerMIL", "TrackerKCF", "createLineSegmentDetector"]

high = ["repeat", "gemm", "solve", "eigen", "dft", "idft", "dct", "idct",
        "warpPolar", "logPolar", "linearPolar", "matchShapes",
        "buildOpticalFlowPyramid", "TrackerCSRT", "TrackerGOTURN",
        "computeCorrespondEpilines", "reprojectImageTo3D",
        "BRISK_create", "colorChange", "illuminationChange"]

missing_critical = [m for mod, m in all_missing if m in critical]
missing_high = [m for mod, m in all_missing if m in high]

print(f"\nCRITICAL ({len(missing_critical)}):", missing_critical if missing_critical else "None!")
print(f"\nHIGH ({len(missing_high)}):", missing_high if missing_high else "None!")
