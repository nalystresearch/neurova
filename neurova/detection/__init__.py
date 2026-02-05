# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Object detection for Neurova"""

from neurova.detection.template import (
    match_template,
    non_max_suppression,
    sliding_window_detection,
    TemplateDetector,
)

from neurova.detection.haar_cascade import HaarCascadeClassifier

# Import HOG detector
from neurova.detection.hog import (
    HOGDescriptor, groupRectangles,
)

__all__ = [
    "match_template",
    "non_max_suppression",
    "sliding_window_detection",
    "TemplateDetector",
    "HaarCascadeClassifier",
    # HOG (NEW)
    "HOGDescriptor",
    "groupRectangles",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.