# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Video processing for Neurova"""

from neurova.video.capture import VideoCapture, VideoWriter
from neurova.video.webcam import WebcamCapture
from neurova.video.optflow import (
    calcOpticalFlowPyrLK, calcOpticalFlowFarneback,
    OPTFLOW_USE_INITIAL_FLOW, OPTFLOW_LK_GET_MIN_EIGENVALS, OPTFLOW_FARNEBACK_GAUSSIAN,
)
from neurova.video.background import (
    BackgroundSubtractor, BackgroundSubtractorMOG2, BackgroundSubtractorKNN,
    createBackgroundSubtractorMOG2, createBackgroundSubtractorKNN,
)
from neurova.video.tracking import (
    CamShift, meanShift, KalmanFilter,
    TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_COUNT,
)
from neurova.video.trackers import (
    Tracker, TrackerMIL, TrackerKCF, TrackerCSRT,
    TrackerMIL_create, TrackerKCF_create, TrackerCSRT_create,
)

__all__ = [
    "VideoCapture",
    "VideoWriter",
    "WebcamCapture",
    
    # Optical flow
    "calcOpticalFlowPyrLK",
    "calcOpticalFlowFarneback",
    "OPTFLOW_USE_INITIAL_FLOW",
    "OPTFLOW_LK_GET_MIN_EIGENVALS",
    "OPTFLOW_FARNEBACK_GAUSSIAN",
    
    # Background subtraction
    "BackgroundSubtractor",
    "BackgroundSubtractorMOG2",
    "BackgroundSubtractorKNN",
    "createBackgroundSubtractorMOG2",
    "createBackgroundSubtractorKNN",
    
    # Tracking
    "CamShift",
    "meanShift",
    "KalmanFilter",
    "TERM_CRITERIA_EPS",
    "TERM_CRITERIA_MAX_ITER",
    "TERM_CRITERIA_COUNT",
    
    # Object Trackers
    "Tracker",
    "TrackerMIL",
    "TrackerKCF",
    "TrackerCSRT",
    "TrackerMIL_create",
    "TrackerKCF_create",
    "TrackerCSRT_create",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.