# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova - advanced image processing and deep learning library

this package provides comprehensive computer vision and deep learning
functionality with minimal dependencies and optional gpu acceleration.
"""

from neurova.version import (
    __version__,
    __version_info__,
    __build__,
    __date__,
    get_version,
    get_build_info,
)

# import device management for gpu and cpu selection
from neurova.device import (
    set_device,
    get_device,
    get_backend,
    cuda_is_available,
    get_device_count,
    get_device_name,
    get_device_info,
    to_device,
    synchronize,
    empty_cache,
    get_memory_usage,
    device_context,
    array,
    zeros,
    ones,
    empty,
)

# import core modules for convenience
from neurova import core
from neurova import io
from neurova import transform
from neurova import filters
from neurova import features
from neurova import neural
from neurova import segmentation
from neurova import detection
from neurova import utils
from neurova import video
from neurova import ml
from neurova import morphology

# neurova compatible modules
from neurova import imgproc
from neurova import highgui
from neurova import dnn

# neurova vision api
from neurova.nvc import VideoCapture
from neurova import nvc

# datasets module
from neurova import data
from neurova.data import datasets
from neurova.data.datasets import (
    load_iris,
    load_titanic,
    load_boston_housing,
    load_diabetes,
    load_air_passengers,
    load_fashion_mnist,
    load_sample_image,
    get_haarcascade,
    list_datasets,
)

# face module
from neurova import face
from neurova.face import (
    FaceDetector,
    FaceRecognizer,
    FaceDataset,
    FaceTrainer,
)

# object detection module - YOLO-style object detection
from neurova import object_detection
from neurova.object_detection import (
    ObjectDetector,
    DetectionDataset,
    DetectionTrainer,
    Detection,
    DetectionResult,
)

# architecture module - pre-built neural network architectures
from neurova import architecture
from neurova.architecture import (
    # Architectures
    MLP,
    CNN,
    LSTM,
    GRU,
    Transformer,
    Autoencoder,
    VAE,
    GAN,
    # Convenience functions
    create_cnn,
    create_rnn,
    create_mlp,
    create_transformer,
    create_autoencoder,
    # Tuning
    GridSearchCV,
    RandomSearchCV,
    BayesianOptimization,
    AutoML,
    HyperparameterSpace,
    tune_model,
)

__all__ = [
    # version info
    "__version__",
    "__version_info__",
    "__build__",
    "__date__",
    "get_version",
    "get_build_info",
    # device management (GPU/CPU selection)
    "set_device",
    "get_device",
    "get_backend",
    "cuda_is_available",
    "get_device_count",
    "get_device_name",
    "get_device_info",
    "to_device",
    "synchronize",
    "empty_cache",
    "get_memory_usage",
    "device_context",
    "array",
    "zeros",
    "ones",
    "empty",
    # modules
    "core",
    "io",
    "transform",
    "filters",
    "features",
    "neural",
    "segmentation",
    "detection",
    "utils",
    "video",
    "VideoCapture",
    "nvc",
    "ml",
    "morphology",
    # compatible modules
    "imgproc",
    "highgui",
    "dnn",
    # datasets
    "data",
    "datasets",
    "load_iris",
    "load_titanic",
    "load_boston_housing",
    "load_diabetes",
    "load_air_passengers",
    "load_fashion_mnist",
    "load_sample_image",
    "get_haarcascade",
    "list_datasets",
    # face module
    "face",
    "FaceDetector",
    "FaceRecognizer",
    "FaceDataset",
    "FaceTrainer",
    # object detection module
    "object_detection",
    "ObjectDetector",
    "DetectionDataset",
    "DetectionTrainer",
    "Detection",
    "DetectionResult",
    # architecture module
    "architecture",
    "MLP",
    "CNN",
    "LSTM",
    "GRU",
    "Transformer",
    "Autoencoder",
    "VAE",
    "GAN",
    "create_cnn",
    "create_rnn",
    "create_mlp",
    "create_transformer",
    "create_autoencoder",
    "GridSearchCV",
    "RandomSearchCV",
    "BayesianOptimization",
    "AutoML",
    "HyperparameterSpace",
    "tune_model",
]

# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.
