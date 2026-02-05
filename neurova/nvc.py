# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""neurova.nvc

pure neurova computer vision apis.

this module provides computer vision functions and classes implemented
entirely in python with numpy.

features:
- video capture from webcam and files
- cascade classifier for object detection
- full color space conversions with cvtcolor
- drawing functions for lines, circles, rectangles, text
- contour detection and analysis
- high gui functions for display and interaction
- deep neural network inference
- image processing filters and transforms
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from neurova.video.webcam import WebcamCapture
from neurova.video.capture import VideoCapture as _FileVideoCapture
from neurova.detection.haar_cascade import HaarCascadeClassifier

# Import full color conversion module
from neurova.imgproc.color import (
    cvtColor,
    # All color constants
    COLOR_BGR2BGRA, COLOR_RGB2RGBA, COLOR_BGRA2BGR, COLOR_RGBA2RGB,
    COLOR_BGR2RGBA, COLOR_RGB2BGRA, COLOR_RGBA2BGR, COLOR_BGRA2RGB,
    COLOR_BGR2RGB, COLOR_RGB2BGR, COLOR_BGRA2RGBA, COLOR_RGBA2BGRA,
    COLOR_BGR2GRAY, COLOR_RGB2GRAY, COLOR_GRAY2BGR, COLOR_GRAY2RGB,
    COLOR_GRAY2BGRA, COLOR_GRAY2RGBA, COLOR_BGRA2GRAY, COLOR_RGBA2GRAY,
    COLOR_BGR2HSV, COLOR_RGB2HSV, COLOR_HSV2BGR, COLOR_HSV2RGB,
    COLOR_BGR2HSV_FULL, COLOR_RGB2HSV_FULL, COLOR_HSV2BGR_FULL, COLOR_HSV2RGB_FULL,
    COLOR_BGR2HLS, COLOR_RGB2HLS, COLOR_HLS2BGR, COLOR_HLS2RGB,
    COLOR_BGR2Lab, COLOR_RGB2Lab, COLOR_Lab2BGR, COLOR_Lab2RGB,
    COLOR_BGR2Luv, COLOR_RGB2Luv, COLOR_Luv2BGR, COLOR_Luv2RGB,
    COLOR_BGR2YCrCb, COLOR_RGB2YCrCb, COLOR_YCrCb2BGR, COLOR_YCrCb2RGB,
    COLOR_BGR2YUV, COLOR_RGB2YUV, COLOR_YUV2BGR, COLOR_YUV2RGB,
    COLOR_BGR2XYZ, COLOR_RGB2XYZ, COLOR_XYZ2BGR, COLOR_XYZ2RGB,
)

# Import drawing functions
from neurova.imgproc.drawing import (
    line, arrowedLine, circle, ellipse,
    polylines, fillPoly, fillConvexPoly, putText, getTextSize,
    drawMarker, drawContours,
    LINE_4, LINE_8, LINE_AA, FILLED,
    FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
    FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
    FONT_HERSHEY_SCRIPT_SIMPLEX, FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC,
    MARKER_CROSS, MARKER_TILTED_CROSS, MARKER_STAR, MARKER_DIAMOND,
    MARKER_SQUARE, MARKER_TRIANGLE_UP, MARKER_TRIANGLE_DOWN,
)

# Import contour functions
from neurova.imgproc.contours import (
    findContours, contourArea, arcLength, boundingRect,
    minAreaRect, minEnclosingCircle, convexHull, approxPolyDP,
    moments, isContourConvex, pointPolygonTest,
    convexityDefects, fitLine, fitEllipse, minEnclosingTriangle,
    RETR_EXTERNAL, RETR_LIST, RETR_CCOMP, RETR_TREE, RETR_FLOODFILL,
    CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE, CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS,
    DIST_USER, DIST_L1 as DIST_L1_FIT, DIST_L2 as DIST_L2_FIT, DIST_C as DIST_C_FIT,
    DIST_L12, DIST_FAIR, DIST_WELSCH, DIST_HUBER,
)

# Import highgui functions
from neurova.highgui import (
    namedWindow, imshow, waitKey, waitKeyEx, destroyWindow, destroyAllWindows,
    moveWindow, resizeWindow, getWindowProperty, setWindowProperty, setWindowTitle,
    createTrackbar, getTrackbarPos, setTrackbarPos, setTrackbarMin, setTrackbarMax,
    setMouseCallback, selectROI, selectROIs,
    WINDOW_NORMAL, WINDOW_AUTOSIZE, WINDOW_OPENGL, WINDOW_FULLSCREEN,
    WINDOW_FREERATIO, WINDOW_KEEPRATIO, WINDOW_GUI_EXPANDED, WINDOW_GUI_NORMAL,
    WND_PROP_FULLSCREEN, WND_PROP_AUTOSIZE, WND_PROP_ASPECT_RATIO,
    WND_PROP_OPENGL, WND_PROP_VISIBLE, WND_PROP_TOPMOST, WND_PROP_VSYNC,
    EVENT_MOUSEMOVE, EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN, EVENT_MBUTTONDOWN,
    EVENT_LBUTTONUP, EVENT_RBUTTONUP, EVENT_MBUTTONUP, EVENT_LBUTTONDBLCLK,
    EVENT_RBUTTONDBLCLK, EVENT_MBUTTONDBLCLK, EVENT_MOUSEWHEEL, EVENT_MOUSEHWHEEL,
    EVENT_FLAG_LBUTTON, EVENT_FLAG_RBUTTON, EVENT_FLAG_MBUTTON,
    EVENT_FLAG_CTRLKEY, EVENT_FLAG_SHIFTKEY, EVENT_FLAG_ALTKEY,
)

# Import DNN module
from neurova import dnn

# Import threshold functions
from neurova.imgproc.threshold import (
    threshold, adaptiveThreshold, inRange,
    THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO,
    THRESH_TOZERO_INV, THRESH_MASK, THRESH_OTSU, THRESH_TRIANGLE,
    ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C,
)

# Import calibration functions
from neurova.calibration import (
    findChessboardCorners, findChessboardCornersSB, cornerSubPix,
    drawChessboardCorners, calibrateCamera, getOptimalNewCameraMatrix,
    undistort, undistortPoints,
    CALIB_CB_ADAPTIVE_THRESH, CALIB_CB_NORMALIZE_IMAGE, CALIB_CB_FILTER_QUADS,
    CALIB_CB_FAST_CHECK, CALIB_CB_EXHAUSTIVE, CALIB_CB_ACCURACY,
    CALIB_USE_INTRINSIC_GUESS, CALIB_FIX_ASPECT_RATIO, CALIB_FIX_PRINCIPAL_POINT,
    CALIB_ZERO_TANGENT_DIST, CALIB_FIX_K1, CALIB_FIX_K2, CALIB_FIX_K3,
    # Pose estimation (NEW)
    solvePnP, solvePnPRansac, projectPoints,
    findHomography, findFundamentalMat, findEssentialMat,
    Rodrigues, decomposeHomographyMat, triangulatePoints,
    SOLVEPNP_ITERATIVE, SOLVEPNP_P3P, SOLVEPNP_AP3P, SOLVEPNP_EPNP,
    SOLVEPNP_DLS, SOLVEPNP_UPNP, SOLVEPNP_IPPE, SOLVEPNP_IPPE_SQUARE, SOLVEPNP_SQPNP,
    RANSAC, LMEDS, RHO,
    FM_7POINT, FM_8POINT, FM_RANSAC, FM_LMEDS,
)

# Import core operations
from neurova.core.ops import (
    add, subtract, multiply, divide, addWeighted, absdiff, convertScaleAbs,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    flip, rotate, split, merge, minMaxLoc, normalize, countNonZero,
    mean, meanStdDev, LUT, copyMakeBorder, magnitude, phase, cartToPolar, polarToCart,
    # New array functions
    hconcat, vconcat, repeat, transpose, reduce,
    REDUCE_SUM, REDUCE_AVG, REDUCE_MAX, REDUCE_MIN,
    # Comparison functions
    inRange as _inRange_ops, compare, checkRange,
    CMP_EQ, CMP_GT, CMP_GE, CMP_LT, CMP_LE, CMP_NE,
    # Math functions
    sqrt, pow as cv_pow, exp, log, min as cv_min, max as cv_max, sum as cv_sum, trace,
    # Linear algebra
    determinant, invert, solve, eigen, SVDecomp, gemm, mulTransposed, completeSymm, setIdentity,
    DECOMP_LU, DECOMP_SVD, DECOMP_EIG, DECOMP_CHOLESKY,
    GEMM_1_T, GEMM_2_T, GEMM_3_T,
    # Existing constants
    FLIP_HORIZONTAL, FLIP_VERTICAL, FLIP_BOTH,
    ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE,
    BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
    BORDER_WRAP, BORDER_REFLECT_101, BORDER_DEFAULT,
    BORDER_TRANSPARENT, BORDER_ISOLATED,
)

# Import Hough transforms
from neurova.imgproc.hough import (
    HoughLines, HoughLinesP, HoughCircles,
    HOUGH_STANDARD, HOUGH_PROBABILISTIC, HOUGH_MULTI_SCALE,
    HOUGH_GRADIENT, HOUGH_GRADIENT_ALT,
)

# Import geometric transforms
from neurova.imgproc.geometric import (
    getPerspectiveTransform, getAffineTransform, getRotationMatrix2D,
    warpPerspective, warpAffine, remap, invertAffineTransform,
    perspectiveTransform, transform,
    INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4,
    INTER_LINEAR_EXACT, INTER_NEAREST_EXACT,
    WARP_FILL_OUTLIERS, WARP_INVERSE_MAP,
)

# Import histogram functions
from neurova.imgproc.histogram import (
    calcHist, compareHist, equalizeHist as _equalizeHist, calcBackProject,
    CLAHE, createCLAHE,
    HISTCMP_CORREL, HISTCMP_CHISQR, HISTCMP_INTERSECT,
    HISTCMP_BHATTACHARYYA, HISTCMP_HELLINGER, HISTCMP_CHISQR_ALT, HISTCMP_KL_DIV,
)

# Import pyramid functions
from neurova.imgproc.pyramid import pyrDown, pyrUp, buildPyramid

# Import connected components
from neurova.imgproc.connected import (
    connectedComponents, connectedComponentsWithStats,
    CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA,
)

# Import optical flow
from neurova.video.optflow import (
    calcOpticalFlowPyrLK, calcOpticalFlowFarneback,
    OPTFLOW_USE_INITIAL_FLOW, OPTFLOW_LK_GET_MIN_EIGENVALS, OPTFLOW_FARNEBACK_GAUSSIAN,
)

# Import background subtraction
from neurova.video.background import (
    BackgroundSubtractor, BackgroundSubtractorMOG2, BackgroundSubtractorKNN,
    createBackgroundSubtractorMOG2, createBackgroundSubtractorKNN,
)

# Import feature descriptors
from neurova.features.descriptors import (
    KeyPoint, Feature2D, ORB, SIFT, AKAZE,
    ORB_create, SIFT_create, AKAZE_create,
    HARRIS_SCORE, FAST_SCORE,
    AKAZE_DESCRIPTOR_KAZE_UPRIGHT, AKAZE_DESCRIPTOR_KAZE,
    AKAZE_DESCRIPTOR_MLDB_UPRIGHT, AKAZE_DESCRIPTOR_MLDB,
)

# Import feature matching
from neurova.features.matching import (
    DMatch, DescriptorMatcher, BFMatcher, FlannBasedMatcher,
    BFMatcher_create, FlannBasedMatcher_create,
    drawKeypoints, drawMatches, drawMatchesKnn,
    NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_HAMMING, NORM_HAMMING2,
    DRAW_MATCHES_FLAGS_DEFAULT, DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS, DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# Import additional feature detectors (NEW)
from neurova.features.detectors import (
    FastFeatureDetector, BRISK, MSER, GFTTDetector, SimpleBlobDetector,
    goodFeaturesToTrack as _goodFeaturesToTrack,
    FAST_FEATURE_DETECTOR_TYPE_5_8, FAST_FEATURE_DETECTOR_TYPE_7_12,
    FAST_FEATURE_DETECTOR_TYPE_9_16,
)

# Import template matching (NEW)
from neurova.imgproc.template import (
    matchTemplate, minMaxLoc as _minMaxLoc,
    TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED,
    TM_CCOEFF, TM_CCOEFF_NORMED,
)

# Import corner detection (NEW)
from neurova.imgproc.corners import (
    cornerHarris, cornerMinEigenVal, cornerEigenValsAndVecs,
    cornerSubPix as _cornerSubPix, preCornerDetect,
)

# Import segmentation functions (NEW)
from neurova.imgproc.segmentation import (
    floodFill, distanceTransform, distanceTransformWithLabels,
    watershed, grabCut, pyrMeanShiftFiltering,
    FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY,
    DIST_L1, DIST_L2, DIST_C, DIST_MASK_3, DIST_MASK_5, DIST_MASK_PRECISE,
    GC_INIT_WITH_RECT, GC_INIT_WITH_MASK, GC_EVAL, GC_EVAL_FREEZE_MODEL,
    GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD,
)

# Import bilateral filtering (NEW)
from neurova.filters.bilateral import (
    bilateralFilter, boxFilter, sqrBoxFilter,
    getGaussianKernel, getGaborKernel, getDerivKernels,
)

# Import HOG detector (NEW)
from neurova.detection.hog import (
    HOGDescriptor, groupRectangles,
)

# Import tracking (NEW)
from neurova.video.tracking import (
    CamShift, meanShift, KalmanFilter,
)

# Import object trackers (NEW)
from neurova.video.trackers import (
    Tracker, TrackerMIL, TrackerKCF, TrackerCSRT,
    TrackerMIL_create, TrackerKCF_create, TrackerCSRT_create,
)

# Import stitching (NEW)
from neurova.stitching import (
    Stitcher, createStitcher,
)

# Import photo module
from neurova.photo import (
    fastNlMeansDenoising, fastNlMeansDenoisingColored,
    inpaint, seamlessClone, denoise_TVL1,
    edgePreservingFilter, detailEnhance, pencilSketch, stylization,
    INPAINT_NS, INPAINT_TELEA,
    NORMAL_CLONE, MIXED_CLONE, MONOCHROME_TRANSFER,
    RECURS_FILTER, NORMCONV_FILTER,
    # HDR and Tonemap (NEW)
    Tonemap, TonemapDrago, TonemapReinhard, TonemapMantiuk,
    createTonemap, createTonemapDrago, createTonemapReinhard, createTonemapMantiuk,
    MergeDebevec, MergeMertens, CalibrateDebevec,
    createMergeDebevec, createMergeMertens, createCalibrateDebevec,
)

# Import I/O functions
from neurova.io import imread, imwrite

# Import filters with aliases
from neurova.filters.blur import (
    gaussian_blur as GaussianBlur,
    box_blur as blur,  # blur() is box blur
    median_blur as medianBlur,
)
from neurova.filters.edges import (
    sobel as Sobel,
    canny as Canny,
    laplacian as Laplacian,
    scharr as Scharr,
)
from neurova.filters import filter2d as filter2D

# Import resize
from neurova.transform.resize import resize

# Import VideoWriter
from neurova.video.capture import VideoWriter

# Import morphology functions
from neurova.morphology import (
    getStructuringElement, erode, dilate, morphologyEx,
    MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE,
    MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE,
    MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT, MORPH_HITMISS,
)

# constants

CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_FRAME_COUNT = 7
CAP_PROP_POS_FRAMES = 1
CAP_PROP_POS_MSEC = 0

CASCADE_SCALE_IMAGE = 0

# Termination criteria
TERM_CRITERIA_EPS = 1
TERM_CRITERIA_MAX_ITER = 2
TERM_CRITERIA_COUNT = TERM_CRITERIA_MAX_ITER


def VideoWriter_fourcc(c1: str, c2: str, c3: str, c4: str) -> int:
    """Create a FourCC code from four characters.
    
    Args:
        c1, c2, c3, c4: Four characters for the codec
    
    Returns:
        FourCC integer code
    """
    return (ord(c1) & 0xFF) | ((ord(c2) & 0xFF) << 8) | \
           ((ord(c3) & 0xFF) << 16) | ((ord(c4) & 0xFF) << 24)


# Alias for nvc.VideoWriter_fourcc
fourcc = VideoWriter_fourcc


def goodFeaturesToTrack(
    image: np.ndarray,
    maxCorners: int,
    qualityLevel: float,
    minDistance: float,
    mask: Optional[np.ndarray] = None,
    blockSize: int = 3,
    useHarrisDetector: bool = False,
    k: float = 0.04
) -> np.ndarray:
    """Find strong corners in an image (Shi-Tomasi or Harris).
    
    Args:
        image: 8-bit or floating-point 32-bit, single-channel image
        maxCorners: Maximum number of corners to return
        qualityLevel: Minimum accepted quality of corners (0-1)
        minDistance: Minimum Euclidean distance between corners
        mask: Optional region of interest
        blockSize: Size of averaging block for derivative computation
        useHarrisDetector: Whether to use Harris detector
        k: Harris detector free parameter
    
    Returns:
        Array of corners (N, 1, 2)
    """
    if image.ndim != 2:
        from .imgproc.color import cvtColor, COLOR_BGR2GRAY
        gray = cvtColor(image, COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # Compute derivatives
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    Ix[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2
    Iy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2
    
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # Box filter (simplified block average)
    from scipy.ndimage import uniform_filter
    Sxx = uniform_filter(Ixx, size=blockSize)
    Sxy = uniform_filter(Ixy, size=blockSize)
    Syy = uniform_filter(Iyy, size=blockSize)
    
    # Compute corner response
    if useHarrisDetector:
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        response = det - k * trace * trace
    else:
        # Shi-Tomasi: minimum eigenvalue
        trace = Sxx + Syy
        det = Sxx * Syy - Sxy * Sxy
        discriminant = np.sqrt(np.maximum(trace * trace - 4 * det, 0))
        response = (trace - discriminant) / 2
    
    # Apply mask
    if mask is not None:
        response = response * (mask > 0)
    
    # Find corners above threshold
    max_response = response.max()
    threshold = max_response * qualityLevel
    
    # Find local maxima
    corners = []
    
    # Get all candidates above threshold
    candidates = np.argwhere(response > threshold)
    
    # Sort by response (strongest first)
    candidate_responses = [(response[y, x], x, y) for y, x in candidates]
    candidate_responses.sort(reverse=True)
    
    # Non-maximum suppression
    for resp, x, y in candidate_responses:
        if len(corners) >= maxCorners:
            break
        
        # Check distance to existing corners
        too_close = False
        for cx, cy in corners:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < minDistance:
                too_close = True
                break
        
        if not too_close:
            corners.append((x, y))
    
    if len(corners) == 0:
        return np.array([], dtype=np.float32).reshape(0, 1, 2)
    
    return np.array(corners, dtype=np.float32).reshape(-1, 1, 2)


def equalizeHist(gray: np.ndarray) -> np.ndarray:
    """Histogram equalization for uint8 grayscale."""
    if gray.ndim != 2:
        raise ValueError("equalizeHist expects a 2D grayscale image")
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8, copy=False)

    hist = np.bincount(gray.ravel(), minlength=256)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return gray

    cdf_min = int(nonzero[0])
    denom = int(cdf[-1] - cdf_min)
    if denom <= 0:
        return gray

    lut = ((cdf - cdf_min) * 255 // denom).astype(np.uint8)
    return lut[gray]


def rectangle(img: np.ndarray, pt1, pt2, color, thickness: int = 1):
    """Draw a rectangle in-place (Neurova drawing function)."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)

    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if img.ndim == 2:
        col = int(color) if not isinstance(color, (tuple, list)) else int(color[0])
    else:
        col = tuple(int(c) for c in color)

    tmax = max(1, int(thickness))
    for t in range(tmax):
        yy1 = max(0, y1 + t)
        yy2 = max(0, y2 - t)
        xx1 = max(0, x1 + t)
        xx2 = max(0, x2 - t)
        img[yy1, xx1 : xx2 + 1] = col
        img[yy2, xx1 : xx2 + 1] = col
        img[yy1 : yy2 + 1, xx1] = col
        img[yy1 : yy2 + 1, xx2] = col
    return img


class CascadeClassifier:
    """Wrapper for Neurova Haar cascades."""

    def __init__(self, filename: Union[str, Path]):
        self._impl = HaarCascadeClassifier(str(filename))

    def empty(self) -> bool:
        return not self._impl.is_loaded

    def detectMultiScale(
        self,
        image: np.ndarray,
        scaleFactor: float = 1.1,
        minNeighbors: int = 5,
        minSize: Tuple[int, int] = (30, 30),
        flags: int = CASCADE_SCALE_IMAGE,
    ):
        dets = self._impl.detect(
            image,
            scale_factor=scaleFactor,
            min_neighbors=minNeighbors,
            min_size=minSize,
        )
        return np.array(
            [(int(x), int(y), int(w), int(h)) for (x, y, w, h, _c) in dets],
            dtype=np.int32,
        )


class VideoCapture:
    """VideoCapture wrapper for webcam and video files.

    - `VideoCapture(int)` opens webcam using Neurova WebcamCapture (ffmpeg-based).
    - `VideoCapture(str/Path/list)` opens a file/sequence using Neurova video.capture.VideoCapture.

    Notes:
    - Frames are returned in RGB order.
    """

    def __init__(self, source=0):
        self._source = source
        self._opened = False
        self._cam: Optional[WebcamCapture] = None
        self._file: Optional[_FileVideoCapture] = None

        self._width = 640
        self._height = 480
        self._fps = 30.0

    def isOpened(self) -> bool:
        return bool(self._opened)

    def open(self, source=None) -> bool:
        if source is not None:
            self._source = source
        if self._opened:
            return True

        if isinstance(self._source, (int, np.integer)):
            self._cam = WebcamCapture(
                device=int(self._source),
                width=int(self._width),
                height=int(self._height),
                fps=float(self._fps),
                pix_fmt="bgr24",
            )
            try:
                ok = self._cam.open()
            except Exception:
                self._cam = None
                self._opened = False
                return False

            self._opened = bool(ok)
            if self._opened:
                # Start background reader for low latency streaming.
                try:
                    self._cam.start()
                except Exception:
                    pass
            return self._opened

        self._file = _FileVideoCapture(self._source, fps=float(self._fps), use_ffmpeg=False)
        self._opened = True
        return True

    def read(self):
        if not self._opened:
            if not self.open():
                return False, None

        if self._cam is not None:
            # First frames may take longer to arrive; keep a modest timeout.
            frame_bgr = self._cam.read_latest(timeout=1.0, copy=False)
            if frame_bgr is None:
                return False, None
            # Already BGR from ffmpeg.
            return True, frame_bgr

        if self._file is not None:
            frame = self._file.read()
            if frame is None:
                return False, None
            # Return RGB frame directly
            return True, frame

        return False, None

    def release(self):
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
        self._cam = None
        self._file = None
        self._opened = False

    def get(self, prop_id: int):
        if prop_id == CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop_id == CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop_id == CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def set(self, prop_id: int, value):
        # ffmpeg capture cannot reliably change settings after open.
        if self._opened:
            return False
        if prop_id == CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
            return True
        if prop_id == CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
            return True
        if prop_id == CAP_PROP_FPS:
            self._fps = float(value)
            return True
        return False


@dataclass(frozen=True)
class _DataPaths:
    haarcascades: str


def _default_haarcascade_dir() -> str:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../nalyst-research
    candidate = repo_root / "example" / "data" / "haarcascades"
    if candidate.exists():
        return str(candidate) + os.sep
    return ""


data = _DataPaths(haarcascades=_default_haarcascade_dir())


# Exports - Neurova API

__all__ = [
    # Classes
    "VideoCapture",
    "CascadeClassifier",
    "data",
    
    # Functions
    "cvtColor",
    "equalizeHist",
    "rectangle",
    "VideoWriter_fourcc",
    "fourcc",
    "goodFeaturesToTrack",
    
    # Drawing functions
    "line", "arrowedLine", "circle", "ellipse",
    "polylines", "fillPoly", "fillConvexPoly", "putText", "getTextSize",
    "drawMarker", "drawContours",
    
    # Contour functions
    "findContours", "contourArea", "arcLength", "boundingRect",
    "minAreaRect", "minEnclosingCircle", "convexHull", "approxPolyDP",
    "moments", "isContourConvex", "pointPolygonTest",
    "convexityDefects", "fitLine", "fitEllipse", "minEnclosingTriangle",
    
    # Thresholding functions
    "threshold", "adaptiveThreshold", "inRange",
    
    # Calibration functions
    "findChessboardCorners", "findChessboardCornersSB", "cornerSubPix",
    "drawChessboardCorners", "calibrateCamera", "getOptimalNewCameraMatrix",
    "undistort", "undistortPoints",
    
    # HighGUI functions
    "namedWindow", "imshow", "waitKey", "waitKeyEx",
    "destroyWindow", "destroyAllWindows",
    "moveWindow", "resizeWindow", "getWindowProperty", "setWindowProperty", "setWindowTitle",
    "createTrackbar", "getTrackbarPos", "setTrackbarPos", "setTrackbarMin", "setTrackbarMax",
    "setMouseCallback", "selectROI", "selectROIs",
    
    # DNN module
    "dnn",
    
    # Core operations (NEW)
    "add", "subtract", "multiply", "divide", "addWeighted", "absdiff", "convertScaleAbs",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "flip", "rotate", "split", "merge", "minMaxLoc", "normalize", "countNonZero",
    "mean", "meanStdDev", "LUT", "copyMakeBorder", "magnitude", "phase", "cartToPolar", "polarToCart",
    # Array operations (NEW)
    "hconcat", "vconcat", "repeat", "transpose", "reduce",
    # Comparison functions (NEW)
    "compare", "checkRange",
    # Math functions (NEW)
    "sqrt", "cv_pow", "exp", "log", "cv_min", "cv_max", "cv_sum", "trace",
    # Linear algebra (NEW)
    "determinant", "invert", "solve", "eigen", "SVDecomp", "gemm", "mulTransposed", "completeSymm", "setIdentity",
    
    # Hough transforms (NEW)
    "HoughLines", "HoughLinesP", "HoughCircles",
    
    # Geometric transforms (NEW)
    "getPerspectiveTransform", "getAffineTransform", "getRotationMatrix2D",
    "warpPerspective", "warpAffine", "remap", "invertAffineTransform",
    "perspectiveTransform", "transform",
    
    # Histogram functions (NEW)
    "calcHist", "compareHist", "calcBackProject", "CLAHE", "createCLAHE",
    
    # Optical flow (NEW)
    "calcOpticalFlowPyrLK", "calcOpticalFlowFarneback",
    
    # Background subtraction (NEW)
    "BackgroundSubtractor", "BackgroundSubtractorMOG2", "BackgroundSubtractorKNN",
    "createBackgroundSubtractorMOG2", "createBackgroundSubtractorKNN",
    
    # Feature detectors (NEW)
    "KeyPoint", "Feature2D", "ORB", "SIFT", "AKAZE",
    "ORB_create", "SIFT_create", "AKAZE_create",
    
    # Feature matching (NEW)
    "DMatch", "DescriptorMatcher", "BFMatcher", "FlannBasedMatcher",
    "BFMatcher_create", "FlannBasedMatcher_create",
    "drawKeypoints", "drawMatches", "drawMatchesKnn",
    
    # Photo module (NEW)
    "fastNlMeansDenoising", "fastNlMeansDenoisingColored",
    "inpaint", "seamlessClone", "denoise_TVL1",
    "edgePreservingFilter", "detailEnhance", "pencilSketch", "stylization",
    
    # I/O functions (NEW)
    "imread", "imwrite",
    
    # Filtering functions (NEW)
    "GaussianBlur", "medianBlur", "blur", "filter2D",
    "Sobel", "Canny", "Laplacian", "Scharr",
    
    # Transform functions (NEW)
    "resize",
    
    # Video writing (NEW)
    "VideoWriter",
    
    # Morphology functions (NEW)
    "getStructuringElement", "erode", "dilate", "morphologyEx",
    
    # Pyramid functions (NEW)
    "pyrDown", "pyrUp", "buildPyramid",
    
    # Connected components (NEW)
    "connectedComponents", "connectedComponentsWithStats",
    
    # Color constants
    "COLOR_BGR2BGRA", "COLOR_RGB2RGBA", "COLOR_BGRA2BGR", "COLOR_RGBA2RGB",
    "COLOR_BGR2RGBA", "COLOR_RGB2BGRA", "COLOR_RGBA2BGR", "COLOR_BGRA2RGB",
    "COLOR_BGR2RGB", "COLOR_RGB2BGR", "COLOR_BGRA2RGBA", "COLOR_RGBA2BGRA",
    "COLOR_BGR2GRAY", "COLOR_RGB2GRAY", "COLOR_GRAY2BGR", "COLOR_GRAY2RGB",
    "COLOR_GRAY2BGRA", "COLOR_GRAY2RGBA", "COLOR_BGRA2GRAY", "COLOR_RGBA2GRAY",
    "COLOR_BGR2HSV", "COLOR_RGB2HSV", "COLOR_HSV2BGR", "COLOR_HSV2RGB",
    "COLOR_BGR2HSV_FULL", "COLOR_RGB2HSV_FULL", "COLOR_HSV2BGR_FULL", "COLOR_HSV2RGB_FULL",
    "COLOR_BGR2HLS", "COLOR_RGB2HLS", "COLOR_HLS2BGR", "COLOR_HLS2RGB",
    "COLOR_BGR2Lab", "COLOR_RGB2Lab", "COLOR_Lab2BGR", "COLOR_Lab2RGB",
    "COLOR_BGR2Luv", "COLOR_RGB2Luv", "COLOR_Luv2BGR", "COLOR_Luv2RGB",
    "COLOR_BGR2YCrCb", "COLOR_RGB2YCrCb", "COLOR_YCrCb2BGR", "COLOR_YCrCb2RGB",
    "COLOR_BGR2YUV", "COLOR_RGB2YUV", "COLOR_YUV2BGR", "COLOR_YUV2RGB",
    "COLOR_BGR2XYZ", "COLOR_RGB2XYZ", "COLOR_XYZ2BGR", "COLOR_XYZ2RGB",
    
    # Threshold types
    "THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO",
    "THRESH_TOZERO_INV", "THRESH_MASK", "THRESH_OTSU", "THRESH_TRIANGLE",
    
    # Adaptive threshold methods
    "ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C",
    
    # Calibration flags
    "CALIB_CB_ADAPTIVE_THRESH", "CALIB_CB_NORMALIZE_IMAGE", "CALIB_CB_FILTER_QUADS",
    "CALIB_CB_FAST_CHECK", "CALIB_CB_EXHAUSTIVE", "CALIB_CB_ACCURACY",
    "CALIB_USE_INTRINSIC_GUESS", "CALIB_FIX_ASPECT_RATIO", "CALIB_FIX_PRINCIPAL_POINT",
    "CALIB_ZERO_TANGENT_DIST", "CALIB_FIX_K1", "CALIB_FIX_K2", "CALIB_FIX_K3",
    
    # Termination criteria
    "TERM_CRITERIA_EPS", "TERM_CRITERIA_MAX_ITER", "TERM_CRITERIA_COUNT",
    
    # Line types
    "LINE_4", "LINE_8", "LINE_AA", "FILLED",
    
    # Font constants
    "FONT_HERSHEY_SIMPLEX", "FONT_HERSHEY_PLAIN", "FONT_HERSHEY_DUPLEX",
    "FONT_HERSHEY_COMPLEX", "FONT_HERSHEY_TRIPLEX", "FONT_HERSHEY_COMPLEX_SMALL",
    "FONT_HERSHEY_SCRIPT_SIMPLEX", "FONT_HERSHEY_SCRIPT_COMPLEX", "FONT_ITALIC",
    
    # Marker types
    "MARKER_CROSS", "MARKER_TILTED_CROSS", "MARKER_STAR", "MARKER_DIAMOND",
    "MARKER_SQUARE", "MARKER_TRIANGLE_UP", "MARKER_TRIANGLE_DOWN",
    
    # Contour retrieval modes
    "RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE", "RETR_FLOODFILL",
    
    # Contour approximation methods
    "CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS",
    
    # Window flags
    "WINDOW_NORMAL", "WINDOW_AUTOSIZE", "WINDOW_OPENGL", "WINDOW_FULLSCREEN",
    "WINDOW_FREERATIO", "WINDOW_KEEPRATIO", "WINDOW_GUI_EXPANDED", "WINDOW_GUI_NORMAL",
    
    # Window properties
    "WND_PROP_FULLSCREEN", "WND_PROP_AUTOSIZE", "WND_PROP_ASPECT_RATIO",
    "WND_PROP_OPENGL", "WND_PROP_VISIBLE", "WND_PROP_TOPMOST", "WND_PROP_VSYNC",
    
    # Mouse events
    "EVENT_MOUSEMOVE", "EVENT_LBUTTONDOWN", "EVENT_RBUTTONDOWN", "EVENT_MBUTTONDOWN",
    "EVENT_LBUTTONUP", "EVENT_RBUTTONUP", "EVENT_MBUTTONUP", "EVENT_LBUTTONDBLCLK",
    "EVENT_RBUTTONDBLCLK", "EVENT_MBUTTONDBLCLK", "EVENT_MOUSEWHEEL", "EVENT_MOUSEHWHEEL",
    "EVENT_FLAG_LBUTTON", "EVENT_FLAG_RBUTTON", "EVENT_FLAG_MBUTTON",
    "EVENT_FLAG_CTRLKEY", "EVENT_FLAG_SHIFTKEY", "EVENT_FLAG_ALTKEY",
    
    # VideoCapture properties
    "CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT", "CAP_PROP_FPS",
    "CAP_PROP_FRAME_COUNT", "CAP_PROP_POS_FRAMES", "CAP_PROP_POS_MSEC",
    
    # Cascade flags
    "CASCADE_SCALE_IMAGE",
    
    # Core operation constants (NEW)
    "FLIP_HORIZONTAL", "FLIP_VERTICAL", "FLIP_BOTH",
    "ROTATE_90_CLOCKWISE", "ROTATE_180", "ROTATE_90_COUNTERCLOCKWISE",
    "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
    "BORDER_WRAP", "BORDER_REFLECT_101", "BORDER_DEFAULT",
    "BORDER_TRANSPARENT", "BORDER_ISOLATED",
    # Reduce constants (NEW)
    "REDUCE_SUM", "REDUCE_AVG", "REDUCE_MAX", "REDUCE_MIN",
    # Comparison constants (NEW)
    "CMP_EQ", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE", "CMP_NE",
    # Decomposition constants (NEW)
    "DECOMP_LU", "DECOMP_SVD", "DECOMP_EIG", "DECOMP_CHOLESKY",
    # GEMM constants (NEW)
    "GEMM_1_T", "GEMM_2_T", "GEMM_3_T",
    
    # Hough constants (NEW)
    "HOUGH_STANDARD", "HOUGH_PROBABILISTIC", "HOUGH_MULTI_SCALE",
    "HOUGH_GRADIENT", "HOUGH_GRADIENT_ALT",
    
    # Interpolation constants (NEW)
    "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "INTER_LANCZOS4",
    "INTER_LINEAR_EXACT", "INTER_NEAREST_EXACT",
    "WARP_FILL_OUTLIERS", "WARP_INVERSE_MAP",
    
    # Histogram constants (NEW)
    "HISTCMP_CORREL", "HISTCMP_CHISQR", "HISTCMP_INTERSECT",
    "HISTCMP_BHATTACHARYYA", "HISTCMP_HELLINGER", "HISTCMP_CHISQR_ALT", "HISTCMP_KL_DIV",
    
    # Optical flow constants (NEW)
    "OPTFLOW_USE_INITIAL_FLOW", "OPTFLOW_LK_GET_MIN_EIGENVALS", "OPTFLOW_FARNEBACK_GAUSSIAN",
    
    # Feature descriptor constants (NEW)
    "HARRIS_SCORE", "FAST_SCORE",
    "AKAZE_DESCRIPTOR_KAZE_UPRIGHT", "AKAZE_DESCRIPTOR_KAZE",
    "AKAZE_DESCRIPTOR_MLDB_UPRIGHT", "AKAZE_DESCRIPTOR_MLDB",
    
    # Feature matching constants (NEW)
    "NORM_INF", "NORM_L1", "NORM_L2", "NORM_L2SQR", "NORM_HAMMING", "NORM_HAMMING2",
    "DRAW_MATCHES_FLAGS_DEFAULT", "DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG",
    "DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS", "DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS",
    
    # Photo constants (NEW)
    "INPAINT_NS", "INPAINT_TELEA",
    "NORMAL_CLONE", "MIXED_CLONE", "MONOCHROME_TRANSFER",
    "RECURS_FILTER", "NORMCONV_FILTER",
    
    # Morphology constants (NEW)
    "MORPH_RECT", "MORPH_CROSS", "MORPH_ELLIPSE",
    "MORPH_ERODE", "MORPH_DILATE", "MORPH_OPEN", "MORPH_CLOSE",
    "MORPH_GRADIENT", "MORPH_TOPHAT", "MORPH_BLACKHAT", "MORPH_HITMISS",
    
    # Connected components constants (NEW)
    "CC_STAT_LEFT", "CC_STAT_TOP", "CC_STAT_WIDTH", "CC_STAT_HEIGHT", "CC_STAT_AREA",
    
    # Pose estimation (NEW)
    "solvePnP", "solvePnPRansac", "projectPoints",
    "findHomography", "findFundamentalMat", "findEssentialMat",
    "Rodrigues", "decomposeHomographyMat", "triangulatePoints",
    "SOLVEPNP_ITERATIVE", "SOLVEPNP_P3P", "SOLVEPNP_AP3P", "SOLVEPNP_EPNP",
    "SOLVEPNP_DLS", "SOLVEPNP_UPNP", "SOLVEPNP_IPPE", "SOLVEPNP_IPPE_SQUARE", "SOLVEPNP_SQPNP",
    "RANSAC", "LMEDS", "RHO",
    "FM_7POINT", "FM_8POINT", "FM_RANSAC", "FM_LMEDS",
    
    # Template matching (NEW)
    "matchTemplate", "TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED",
    "TM_CCOEFF", "TM_CCOEFF_NORMED",
    
    # Corner detection (NEW)
    "cornerHarris", "cornerMinEigenVal", "cornerEigenValsAndVecs", "preCornerDetect",
    
    # Segmentation (NEW)
    "floodFill", "distanceTransform", "distanceTransformWithLabels",
    "watershed", "grabCut", "pyrMeanShiftFiltering",
    "FLOODFILL_FIXED_RANGE", "FLOODFILL_MASK_ONLY",
    "DIST_L1", "DIST_L2", "DIST_C", "DIST_MASK_3", "DIST_MASK_5", "DIST_MASK_PRECISE",
    "GC_INIT_WITH_RECT", "GC_INIT_WITH_MASK", "GC_EVAL", "GC_EVAL_FREEZE_MODEL",
    "GC_BGD", "GC_FGD", "GC_PR_BGD", "GC_PR_FGD",
    
    # Bilateral filtering (NEW)
    "bilateralFilter", "boxFilter", "sqrBoxFilter",
    "getGaussianKernel", "getGaborKernel", "getDerivKernels",
    
    # Feature detectors (NEW)
    "FastFeatureDetector", "BRISK", "MSER", "GFTTDetector", "SimpleBlobDetector",
    "FAST_FEATURE_DETECTOR_TYPE_5_8", "FAST_FEATURE_DETECTOR_TYPE_7_12",
    "FAST_FEATURE_DETECTOR_TYPE_9_16",
    
    # HOG descriptor (NEW)
    "HOGDescriptor", "groupRectangles",
    
    # Tracking (NEW)
    "CamShift", "meanShift", "KalmanFilter",
    
    # Object Trackers (NEW)
    "Tracker", "TrackerMIL", "TrackerKCF", "TrackerCSRT",
    "TrackerMIL_create", "TrackerKCF_create", "TrackerCSRT_create",
    
    # Stitching (NEW)
    "Stitcher", "createStitcher",
    
    # HDR and Tonemap (NEW)
    "Tonemap", "TonemapDrago", "TonemapReinhard", "TonemapMantiuk",
    "createTonemap", "createTonemapDrago", "createTonemapReinhard", "createTonemapMantiuk",
    "MergeDebevec", "MergeMertens", "CalibrateDebevec",
    "createMergeDebevec", "createMergeMertens", "createCalibrateDebevec",
]

# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
