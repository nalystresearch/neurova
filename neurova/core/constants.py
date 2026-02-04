# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Constants and enumerations for Neurova library"""

import math
from enum import Enum, IntEnum


# mathematical constants
PI = math.pi
TAU = 2 * PI
E = math.e
SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)


class ColorSpace(Enum):
    """Supported color spaces"""
    GRAY = "GRAY"
    RGB = "RGB"
    BGR = "BGR"
    RGBA = "RGBA"
    BGRA = "BGRA"
    HSV = "HSV"
    HSL = "HSL"
    LAB = "LAB"
    LUV = "LUV"
    YCRCB = "YCrCb"
    XYZ = "XYZ"


class BorderMode(IntEnum):
    """Border extrapolation methods"""
    CONSTANT = 0  # Fill with constant value
    REPLICATE = 1  # Replicate edge pixels
    REFLECT = 2  # Reflect at border (abcdefg -> fedcba)
    WRAP = 3  # Wrap around (abcdefg -> cdefgab)
    REFLECT_101 = 4  # Reflect without repeating edge (abcdefg -> gfedcb)
    DEFAULT = REFLECT_101


class InterpolationMode(IntEnum):
    """Interpolation methods for resizing and warping"""
    NEAREST = 0  # Nearest neighbor
    LINEAR = 1  # Bilinear interpolation
    CUBIC = 2  # Bicubic interpolation
    LANCZOS = 3  # Lanczos resampling
    AREA = 4  # Area-based resampling


class ThresholdMethod(Enum):
    """Thresholding methods"""
    BINARY = "BINARY"
    BINARY_INV = "BINARY_INV"
    TRUNCATE = "TRUNCATE"
    TO_ZERO = "TO_ZERO"
    TO_ZERO_INV = "TO_ZERO_INV"
    OTSU = "OTSU"
    ADAPTIVE_MEAN = "ADAPTIVE_MEAN"
    ADAPTIVE_GAUSSIAN = "ADAPTIVE_GAUSSIAN"


class MorphologyOp(Enum):
    """Morphological operations"""
    ERODE = "ERODE"
    DILATE = "DILATE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    GRADIENT = "GRADIENT"
    TOPHAT = "TOPHAT"
    BLACKHAT = "BLACKHAT"


class KernelShape(Enum):
    """Structuring element shapes"""
    RECT = "RECT"
    ELLIPSE = "ELLIPSE"
    CROSS = "CROSS"


class EdgeDetectionMethod(Enum):
    """Edge detection methods"""
    SOBEL = "SOBEL"
    SCHARR = "SCHARR"
    LAPLACIAN = "LAPLACIAN"
    CANNY = "CANNY"
    PREWITT = "PREWITT"
    ROBERTS = "ROBERTS"


class CornerDetectionMethod(Enum):
    """Corner detection methods"""
    HARRIS = "HARRIS"
    SHI_TOMASI = "SHI_TOMASI"
    FAST = "FAST"
    GOOD_FEATURES = "GOOD_FEATURES"


class DescriptorType(Enum):
    """Feature descriptor types"""
    ORB = "ORB"
    BRIEF = "BRIEF"
    BRISK = "BRISK"
    FREAK = "FREAK"


class MatcherType(Enum):
    """Feature matcher types"""
    BRUTE_FORCE = "BRUTE_FORCE"
    FLANN = "FLANN"


class DistanceMetric(Enum):
    """Distance metrics"""
    L1 = "L1"  # Manhattan distance
    L2 = "L2"  # Euclidean distance
    HAMMING = "HAMMING"  # Hamming distance
    COSINE = "COSINE"  # Cosine distance
    CORRELATION = "CORRELATION"  # Correlation


class ActivationFunction(Enum):
    """Neural network activation functions"""
    RELU = "RELU"
    LEAKY_RELU = "LEAKY_RELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    SOFTMAX = "SOFTMAX"
    SWISH = "SWISH"
    GELU = "GELU"
    ELU = "ELU"
    SELU = "SELU"
    LINEAR = "LINEAR"


class LossFunction(Enum):
    """Loss functions"""
    MSE = "MSE"  # Mean Squared Error
    MAE = "MAE"  # Mean Absolute Error
    CROSS_ENTROPY = "CROSS_ENTROPY"
    BINARY_CROSS_ENTROPY = "BINARY_CROSS_ENTROPY"
    CATEGORICAL_CROSS_ENTROPY = "CATEGORICAL_CROSS_ENTROPY"
    FOCAL = "FOCAL"
    DICE = "DICE"
    IOU = "IOU"


class OptimizerType(Enum):
    """Optimizer types"""
    SGD = "SGD"
    MOMENTUM = "MOMENTUM"
    ADAM = "ADAM"
    ADAMW = "ADAMW"
    RMSPROP = "RMSPROP"
    ADAGRAD = "ADAGRAD"


class PaddingMode(Enum):
    """Padding modes"""
    ZEROS = "ZEROS"
    SAME = "SAME"
    VALID = "VALID"
    REFLECT = "REFLECT"
    REPLICATE = "REPLICATE"


class PoolingType(Enum):
    """Pooling operation types"""
    MAX = "MAX"
    AVG = "AVG"
    GLOBAL_MAX = "GLOBAL_MAX"
    GLOBAL_AVG = "GLOBAL_AVG"


# default values
DEFAULT_BORDER_MODE = BorderMode.REFLECT_101
DEFAULT_INTERPOLATION = InterpolationMode.LINEAR
DEFAULT_THRESHOLD = 127
DEFAULT_KERNEL_SIZE = 3
DEFAULT_SIGMA = 1.0

# numeric limits
EPSILON = 1e-7
MAX_PIXEL_VALUE_UINT8 = 255
MAX_PIXEL_VALUE_UINT16 = 65535
MAX_PIXEL_VALUE_FLOAT = 1.0

# image formats
SUPPORTED_IMAGE_FORMATS = {
    "jpeg", "jpg", "png", "bmp", "tiff", "tif", "webp", "ppm", "pgm", "pbm"
}

SUPPORTED_VIDEO_FORMATS = {
    "mp4", "avi", "mov", "mkv", "flv", "wmv", "webm"
}

# cascade classifier types
CASCADE_FACE_DEFAULT = "haarcascade_frontalface_default"
CASCADE_FACE_ALT = "haarcascade_frontalface_alt"
CASCADE_EYE = "haarcascade_eye"
CASCADE_SMILE = "haarcascade_smile"

# color conversion codes
RGB_TO_GRAY_WEIGHTS = (0.299, 0.587, 0.114)  # Standard luminance weights
BGR_TO_GRAY_WEIGHTS = (0.114, 0.587, 0.299)  # BGR order
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.