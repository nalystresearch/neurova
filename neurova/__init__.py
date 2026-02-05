# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

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

# import smart c++ loader for acceleration
from neurova._cpp_loader import get_cpp_status, CPPModuleLoader

# initialize c++ module loader
_cpp_loader = CPPModuleLoader()

# try to load c++ accelerated modules first, fallback to python
try:
    # load minimal bundle (imgproc, filters, features)
    from neurova import neurova_minimal
    imgproc = neurova_minimal
    filters = neurova_minimal
    features = neurova_minimal
    _cpp_loader.cpp_modules['minimal'] = True
except ImportError:
    from neurova import imgproc
    from neurova import filters
    from neurova import features
    _cpp_loader.fallback_modules['minimal'] = True

try:
    # load architecture bundle (neural network architectures)
    from neurova import neurova_architecture
    architecture = neurova_architecture
    _cpp_loader.cpp_modules['architecture'] = True
except ImportError:
    from neurova import architecture
    _cpp_loader.fallback_modules['architecture'] = True

try:
    # load extended bundle (core, augmentation, calibration)
    from neurova import neurova_extended
    core = neurova_extended
    _cpp_loader.cpp_modules['extended'] = True
except ImportError:
    from neurova import core
    _cpp_loader.fallback_modules['extended'] = True

try:
    # load mega bundle (morphology, neural, nn, object_detection)
    from neurova import neurova_mega
    morphology = neurova_mega
    neural = neurova_mega
    nn = neurova_mega
    object_detection = neurova_mega
    _cpp_loader.cpp_modules['mega'] = True
except ImportError:
    from neurova import morphology
    from neurova import neural
    from neurova import nn
    from neurova import object_detection
    _cpp_loader.fallback_modules['mega'] = True

try:
    # load advanced bundle (photo, segmentation, solutions)
    from neurova import neurova_advanced
    segmentation = neurova_advanced
    _cpp_loader.cpp_modules['advanced'] = True
except ImportError:
    from neurova import segmentation
    _cpp_loader.fallback_modules['advanced'] = True

try:
    # load final bundle (utils, transform, stitching)
    from neurova import neurova_final
    utils = neurova_final
    transform = neurova_final
    _cpp_loader.cpp_modules['final'] = True
except ImportError:
    from neurova import utils
    from neurova import transform
    _cpp_loader.fallback_modules['final'] = True

try:
    # load timeseries bundle
    from neurova import neurova_timeseries
    timeseries = neurova_timeseries
    _cpp_loader.cpp_modules['timeseries'] = True
except ImportError:
    from neurova import timeseries
    _cpp_loader.fallback_modules['timeseries'] = True

# python-only modules (intentionally not compiled)
from neurova import io
from neurova import detection
from neurova import video
from neurova import ml
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

# object detection module - single-stage object detection
if 'mega' not in _cpp_loader.cpp_modules:
    # only import python version if c++ didn't load
    from neurova.object_detection import (
        ObjectDetector,
        DetectionDataset,
        DetectionTrainer,
        Detection,
        DetectionResult,
    )
else:
    # use c++ version
    ObjectDetector = object_detection.ObjectDetector if hasattr(object_detection, 'ObjectDetector') else None
    DetectionDataset = object_detection.DetectionDataset if hasattr(object_detection, 'DetectionDataset') else None
    DetectionTrainer = object_detection.DetectionTrainer if hasattr(object_detection, 'DetectionTrainer') else None
    Detection = object_detection.Detection if hasattr(object_detection, 'Detection') else None
    DetectionResult = object_detection.DetectionResult if hasattr(object_detection, 'DetectionResult') else None

# architecture module - pre-built neural network architectures
if 'architecture' not in _cpp_loader.cpp_modules:
    # only import python version if c++ didn't load
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
else:
    # use c++ version
    MLP = architecture.MLP if hasattr(architecture, 'MLP') else None
    CNN = architecture.CNN if hasattr(architecture, 'CNN') else None
    LSTM = architecture.LSTM if hasattr(architecture, 'LSTM') else None
    GRU = architecture.GRU if hasattr(architecture, 'GRU') else None
    Transformer = architecture.Transformer if hasattr(architecture, 'Transformer') else None
    Autoencoder = architecture.Autoencoder if hasattr(architecture, 'Autoencoder') else None
    VAE = architecture.VAE if hasattr(architecture, 'VAE') else None
    GAN = architecture.GAN if hasattr(architecture, 'GAN') else None
    create_cnn = architecture.create_cnn if hasattr(architecture, 'create_cnn') else None
    create_rnn = architecture.create_rnn if hasattr(architecture, 'create_rnn') else None
    create_mlp = architecture.create_mlp if hasattr(architecture, 'create_mlp') else None
    create_transformer = architecture.create_transformer if hasattr(architecture, 'create_transformer') else None
    create_autoencoder = architecture.create_autoencoder if hasattr(architecture, 'create_autoencoder') else None
    GridSearchCV = architecture.GridSearchCV if hasattr(architecture, 'GridSearchCV') else None
    RandomSearchCV = architecture.RandomSearchCV if hasattr(architecture, 'RandomSearchCV') else None
    BayesianOptimization = architecture.BayesianOptimization if hasattr(architecture, 'BayesianOptimization') else None
    AutoML = architecture.AutoML if hasattr(architecture, 'AutoML') else None
    HyperparameterSpace = architecture.HyperparameterSpace if hasattr(architecture, 'HyperparameterSpace') else None
    tune_model = architecture.tune_model if hasattr(architecture, 'tune_model') else None

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
    # modules (may be c++ accelerated)
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
    # c++ loader utilities
    "get_cpp_status",
    "_cpp_loader",
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

# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
