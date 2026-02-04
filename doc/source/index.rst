.. Neurova documentation master file

Neurova Documentation
=====================

**Neurova** is a high-performance computer vision and deep learning library
designed as a modern, GPU-accelerated computer vision library. It provides
comprehensive image processing, video I/O, and neural network inference
capabilities with a focus on ease of use and cross-platform compatibility.

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Getting Started
        :link: getting_started
        :link-type: doc

        New to Neurova? Start here with installation instructions
        and your first steps.

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        Complete API documentation for all Neurova modules.

    .. grid-item-card:: Tutorials
        :link: tutorials/index
        :link-type: doc

        Step-by-step guides for common computer vision tasks.

    .. grid-item-card:: Examples
        :link: examples/index
        :link-type: doc

        Working code examples and sample applications.

Key Features
------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Feature
     - Description
   * - **GPU Acceleration**
     - CUDA, OpenCL, Metal, and Vulkan backends for maximum performance
   * - **Cross-Platform**
     - Windows, macOS, Linux with native GUI support
   * - **Deep Learning**
     - Load and run ONNX and Neurova models
   * - **Image Processing**
     - Complete set of filters, transforms, and analysis tools
   * - **Video I/O**
     - Capture from cameras, files, and network streams
   * - **Neurova Native**
     - Native API with comprehensive documentation

Quick Example
-------------

.. code-block:: python

   import neurova as nv

   # Read an image
   img = nv.imread("photo.jpg")

   # Convert to grayscale
   gray = nv.cvtColor(img, nv.COLOR_BGR2GRAY)

   # Apply Gaussian blur
   blurred = nv.GaussianBlur(gray, (5, 5), 1.5)

   # Detect edges
   edges = nv.Canny(blurred, 50, 150)

   # Display result
   nv.imshow("Edges", edges)
   nv.waitKey(0)

Deep Learning Example
---------------------

.. code-block:: python

   import neurova as nv

   # Load a YOLOv8 model
   net = nv.dnn.readNet("yolov8n.onnx")
   net.setPreferableBackend(nv.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(nv.dnn.DNN_TARGET_CUDA)

   # Prepare input
   img = nv.imread("image.jpg")
   blob = nv.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True)

   # Run inference
   net.setInput(blob)
   output = net.forward()

   # Process detections...

Installation
------------

Install via pip:

.. code-block:: bash

   pip install neurova

Or with GPU support:

.. code-block:: bash

   pip install neurova[cuda]     # For NVIDIA GPUs
   pip install neurova[metal]    # For Apple Silicon
   pip install neurova[opencl]   # For OpenCL devices

Build from source:

.. code-block:: bash

   git clone https://github.com/neurova/neurova.git
   cd neurova
   pip install -e .[dev]

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   installation
   performance_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/core
   api/imgproc
   api/imgcodecs
   api/videoio
   api/highgui
   api/dnn
   api/features2d
   api/calib3d
   api/objdetect
   api/ml
   api/cuda

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/image_basics
   tutorials/video_capture
   tutorials/face_detection
   tutorials/object_detection
   tutorials/custom_training

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index
   examples/image_processing
   examples/deep_learning
   examples/video_analysis
   examples/gpu_computing

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog
   roadmap
   license

Module Overview
---------------

Core Modules
~~~~~~~~~~~~

:doc:`api/core`
    Fundamental data structures (Mat, Point, Rect, Scalar) and array operations.

:doc:`api/imgproc`
    Image processing functions: filtering, color conversion, geometric transforms,
    morphological operations, contour analysis, drawing.

:doc:`api/imgcodecs`
    Image file I/O supporting JPEG, PNG, TIFF, WebP, AVIF, EXR formats.

:doc:`api/videoio`
    Video capture from cameras, files, and network streams. Video writing.

:doc:`api/highgui`
    GUI functions: windows, trackbars, mouse events, keyboard input.

Deep Learning
~~~~~~~~~~~~~

:doc:`api/dnn`
    Deep neural network inference supporting ONNX, Caffe, Darknet formats
    formats with CUDA, OpenCL, Core ML, and Vulkan backends.

Computer Vision
~~~~~~~~~~~~~~~

:doc:`api/features2d`
    Feature detection (ORB, SIFT, AKAZE) and matching.

:doc:`api/calib3d`
    Camera calibration, stereo vision, and 3D reconstruction.

:doc:`api/objdetect`
    Object detection with cascade classifiers and HOG descriptors.

Machine Learning
~~~~~~~~~~~~~~~~

:doc:`api/ml`
    Classical ML algorithms: SVM, k-NN, Decision Trees, Random Forests, Boosting.

GPU Computing
~~~~~~~~~~~~~

:doc:`api/cuda`
    CUDA-specific functions and GPU memory management.

Platform Support
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Platform
     - CPU
     - CUDA
     - OpenCL
     - Metal/Vulkan
   * - Windows x64
     - 
     - 
     - 
     - Vulkan
   * - macOS x64
     - 
     - 
     - 
     - Metal
   * - macOS ARM64
     - 
     - 
     - 
     - Metal
   * - Linux x64
     - 
     - 
     - 
     - Vulkan
   * - Linux ARM64
     - 
     -  (Jetson)
     - 
     - Vulkan

Getting Help
------------

- **Issue Tracker**: https://github.com/neurova/neurova/issues
- **Discussions**: https://github.com/neurova/neurova/discussions
- **Stack Overflow**: Tag your questions with ``neurova``

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
