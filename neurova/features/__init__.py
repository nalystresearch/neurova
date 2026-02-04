# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Feature detection and description for Neurova"""

from neurova.features import corners, keypoints
from neurova.features.corners import detect_corners, harris_response, shi_tomasi_response
from neurova.features.keypoints import detect_keypoints
from neurova.features.types import Keypoint

# Feature descriptors
from neurova.features.descriptors import (
    KeyPoint, Feature2D, ORB, SIFT, AKAZE,
    ORB_create, SIFT_create, AKAZE_create,
    HARRIS_SCORE, FAST_SCORE,
    AKAZE_DESCRIPTOR_KAZE_UPRIGHT, AKAZE_DESCRIPTOR_KAZE,
    AKAZE_DESCRIPTOR_MLDB_UPRIGHT, AKAZE_DESCRIPTOR_MLDB,
)

# Feature matching
from neurova.features.matching import (
    DMatch, DescriptorMatcher, BFMatcher, FlannBasedMatcher,
    BFMatcher_create, FlannBasedMatcher_create,
    drawKeypoints, drawMatches, drawMatchesKnn,
    NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_HAMMING, NORM_HAMMING2,
    DRAW_MATCHES_FLAGS_DEFAULT, DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
    DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS, DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# Additional feature detectors (NEW)
from neurova.features.detectors import (
    FastFeatureDetector, BRISK, MSER, GFTTDetector, SimpleBlobDetector,
    goodFeaturesToTrack,
    FAST_FEATURE_DETECTOR_TYPE_5_8, FAST_FEATURE_DETECTOR_TYPE_7_12,
    FAST_FEATURE_DETECTOR_TYPE_9_16,
)

__all__ = [
    # Original
    "corners",
    "keypoints",
    "Keypoint",
    "detect_corners",
    "detect_keypoints",
    "harris_response",
    "shi_tomasi_response",
    
    # Feature descriptors
    "KeyPoint",
    "Feature2D",
    "ORB", "SIFT", "AKAZE",
    "ORB_create", "SIFT_create", "AKAZE_create",
    "HARRIS_SCORE", "FAST_SCORE",
    "AKAZE_DESCRIPTOR_KAZE_UPRIGHT", "AKAZE_DESCRIPTOR_KAZE",
    "AKAZE_DESCRIPTOR_MLDB_UPRIGHT", "AKAZE_DESCRIPTOR_MLDB",
    
    # Feature matching
    "DMatch",
    "DescriptorMatcher",
    "BFMatcher", "FlannBasedMatcher",
    "BFMatcher_create", "FlannBasedMatcher_create",
    "drawKeypoints", "drawMatches", "drawMatchesKnn",
    "NORM_INF", "NORM_L1", "NORM_L2", "NORM_L2SQR", "NORM_HAMMING", "NORM_HAMMING2",
    "DRAW_MATCHES_FLAGS_DEFAULT", "DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG",
    "DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS", "DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS",
    
    # Additional feature detectors
    "FastFeatureDetector", "BRISK", "MSER", "GFTTDetector", "SimpleBlobDetector",
    "goodFeaturesToTrack",
    "FAST_FEATURE_DETECTOR_TYPE_5_8", "FAST_FEATURE_DETECTOR_TYPE_7_12",
    "FAST_FEATURE_DETECTOR_TYPE_9_16",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.