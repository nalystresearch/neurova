Getting Started with Neurova
=============================

This guide will help you get started with Neurova, a high-performance
computer vision library designed as a modern computer vision library.

Installation
------------

From PyPI
~~~~~~~~~

The easiest way to install Neurova is via pip:

.. code-block:: bash

   pip install neurova

To install with GPU support, use the appropriate extras:

.. code-block:: bash

   # For NVIDIA GPUs (CUDA)
   pip install neurova[cuda]
   
   # For Apple Silicon (Metal)
   pip install neurova[metal]
   
   # For OpenCL devices
   pip install neurova[opencl]
   
   # All GPU backends
   pip install neurova[gpu]

From Source
~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   git clone https://github.com/neurova/neurova.git
   cd neurova
   pip install -e .[dev]

Building with CMake
~~~~~~~~~~~~~~~~~~~

For C++ development or custom builds:

.. code-block:: bash

   git clone https://github.com/neurova/neurova.git
   cd neurova
   mkdir build && cd build
   
   # Configure with desired options
   cmake .. \
       -DCMAKE_BUILD_TYPE=Release \
       -DNEUROVA_WITH_CUDA=ON \
       -DNEUROVA_WITH_OPENCL=ON \
       -DNEUROVA_BUILD_PYTHON=ON
   
   # Build
   cmake --build . --config Release -j$(nproc)
   
   # Install
   sudo cmake --install .

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Print version
   print(f"Neurova version: {nv.__version__}")
   
   # Check available backends
   print(f"CUDA available: {nv.cuda.isAvailable()}")
   print(f"OpenCL available: {nv.opencl.isAvailable()}")
   print(f"Metal available: {nv.metal.isAvailable()}")

Quick Start
-----------

Loading and Displaying an Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Load an image
   img = nv.imread("photo.jpg")
   
   # Check if loaded successfully
   if img is None:
       print("Error loading image")
       exit()
   
   # Print image properties
   print(f"Shape: {img.shape}")      # (height, width, channels)
   print(f"Dtype: {img.dtype}")      # Data type
   print(f"Size: {img.size}")        # Total elements
   
   # Display the image
   nv.imshow("Image", img)
   nv.waitKey(0)  # Wait for key press
   nv.destroyAllWindows()

Basic Image Processing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Load image
   img = nv.imread("photo.jpg")
   
   # Convert to grayscale
   gray = nv.cvtColor(img, nv.COLOR_BGR2GRAY)
   
   # Apply Gaussian blur
   blurred = nv.GaussianBlur(gray, (5, 5), 1.5)
   
   # Detect edges using Canny
   edges = nv.Canny(blurred, 50, 150)
   
   # Display results
   nv.imshow("Original", img)
   nv.imshow("Grayscale", gray)
   nv.imshow("Blurred", blurred)
   nv.imshow("Edges", edges)
   
   nv.waitKey(0)
   nv.destroyAllWindows()
   
   # Save result
   nv.imwrite("edges.png", edges)

Video Capture
~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Open camera (0 = default camera)
   cap = nv.VideoCapture(0)
   
   if not cap.isOpened():
       print("Cannot open camera")
       exit()
   
   while True:
       # Read frame
       ret, frame = cap.read()
       
       if not ret:
           print("Can't receive frame")
           break
       
       # Process frame (e.g., convert to grayscale)
       gray = nv.cvtColor(frame, nv.COLOR_BGR2GRAY)
       
       # Display
       nv.imshow("Camera", gray)
       
       # Exit on 'q' key
       if nv.waitKey(1) & 0xFF == ord('q'):
           break
   
   # Cleanup
   cap.release()
   nv.destroyAllWindows()

Drawing on Images
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Create a blank image
   img = nv.zeros((480, 640, 3), dtype=nv.uint8)
   
   # Draw a line
   nv.line(img, (0, 0), (640, 480), (0, 255, 0), 2)
   
   # Draw a rectangle
   nv.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)
   
   # Draw a filled circle
   nv.circle(img, (320, 240), 50, (255, 0, 0), -1)
   
   # Draw an ellipse
   nv.ellipse(img, (400, 300), (60, 30), 45, 0, 360, (255, 255, 0), 2)
   
   # Add text
   nv.putText(img, "Hello Neurova!", (50, 450),
              nv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
   
   # Display
   nv.imshow("Drawing", img)
   nv.waitKey(0)

Deep Learning Inference
-----------------------

Loading and Running a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Load ONNX model
   net = nv.dnn.readNet("model.onnx")
   
   # Configure backend (optional - uses best available)
   # net.setPreferableBackend(nv.dnn.DNN_BACKEND_CUDA)
   # net.setPreferableTarget(nv.dnn.DNN_TARGET_CUDA)
   
   # Load and preprocess image
   img = nv.imread("input.jpg")
   blob = nv.dnn.blobFromImage(
       img,
       scalefactor=1/255.0,
       size=(224, 224),
       mean=(0.485, 0.456, 0.406),
       swapRB=True,
       crop=False
   )
   
   # Run inference
   net.setInput(blob)
   output = net.forward()
   
   # Process output
   class_id = output.argmax()
   confidence = output[0, class_id]
   print(f"Class: {class_id}, Confidence: {confidence:.2%}")

Object Detection with YOLO
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   import numpy as np
   
   # Load YOLOv8 model
   net = nv.dnn.readNet("yolov8n.onnx")
   
   # Load image
   img = nv.imread("image.jpg")
   height, width = img.shape[:2]
   
   # Prepare input
   blob = nv.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True)
   
   # Inference
   net.setInput(blob)
   outputs = net.forward(net.getUnconnectedOutLayersNames())
   
   # Post-process
   boxes, confidences, class_ids = [], [], []
   
   for output in outputs:
       for detection in output[0]:
           scores = detection[4:]
           class_id = scores.argmax()
           confidence = scores[class_id]
           
           if confidence > 0.5:
               cx, cy, w, h = detection[:4]
               x = int((cx - w/2) * width / 640)
               y = int((cy - h/2) * height / 640)
               w = int(w * width / 640)
               h = int(h * height / 640)
               
               boxes.append([x, y, w, h])
               confidences.append(float(confidence))
               class_ids.append(class_id)
   
   # Apply NMS
   indices = nv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
   
   # Draw results
   for i in indices:
       x, y, w, h = boxes[i]
       nv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
   
   nv.imshow("Detections", img)
   nv.waitKey(0)

Face Detection
~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Create face detector
   detector = nv.dnn.FaceDetectorYN.create(
       model="yunet.onnx",
       config="",
       input_size=(320, 320),
       score_threshold=0.9,
       nms_threshold=0.3
   )
   
   # Load image
   img = nv.imread("photo.jpg")
   
   # Set input size
   detector.setInputSize((img.shape[1], img.shape[0]))
   
   # Detect faces
   _, faces = detector.detect(img)
   
   # Draw results
   if faces is not None:
       for face in faces:
           x, y, w, h = face[:4].astype(int)
           confidence = face[14]
           
           # Draw box
           nv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
           
           # Draw landmarks
           landmarks = face[4:14].reshape(5, 2).astype(int)
           for lm in landmarks:
               nv.circle(img, tuple(lm), 2, (0, 0, 255), -1)
   
   nv.imshow("Faces", img)
   nv.waitKey(0)

GPU Acceleration
----------------

Using CUDA
~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Check CUDA availability
   if not nv.cuda.isAvailable():
       print("CUDA not available")
       exit()
   
   # Print device info
   print(f"CUDA devices: {nv.cuda.getCudaEnabledDeviceCount()}")
   nv.cuda.printShortCudaDeviceInfo(0)
   
   # Load image to CPU
   img_cpu = nv.imread("large_image.jpg")
   
   # Upload to GPU
   img_gpu = nv.cuda.GpuMat()
   img_gpu.upload(img_cpu)
   
   # GPU operations
   gray_gpu = nv.cuda.GpuMat()
   nv.cuda.cvtColor(img_gpu, gray_gpu, nv.COLOR_BGR2GRAY)
   
   blurred_gpu = nv.cuda.GpuMat()
   nv.cuda.GaussianBlur(gray_gpu, blurred_gpu, (5, 5), 1.5)
   
   # Download result to CPU
   result_cpu = blurred_gpu.download()
   
   # Display
   nv.imshow("GPU Result", result_cpu)
   nv.waitKey(0)

DNN with CUDA Backend
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Load model with CUDA backend
   net = nv.dnn.readNet("model.onnx")
   net.setPreferableBackend(nv.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(nv.dnn.DNN_TARGET_CUDA)
   
   # For half-precision (faster on RTX GPUs)
   # net.setPreferableTarget(nv.dnn.DNN_TARGET_CUDA_FP16)
   
   # Run inference (same as CPU)
   blob = nv.dnn.blobFromImage(img, 1/255.0, (640, 640))
   net.setInput(blob)
   output = net.forward()

Common Tasks
------------

Image Resizing
~~~~~~~~~~~~~~

.. code-block:: python

   # Resize to specific dimensions
   resized = nv.resize(img, (640, 480))
   
   # Resize by scale factor
   scaled = nv.resize(img, None, fx=0.5, fy=0.5)
   
   # Different interpolation methods
   nearest = nv.resize(img, (640, 480), interpolation=nv.INTER_NEAREST)
   cubic = nv.resize(img, (640, 480), interpolation=nv.INTER_CUBIC)
   lanczos = nv.resize(img, (640, 480), interpolation=nv.INTER_LANCZOS4)

Color Manipulation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Color space conversions
   gray = nv.cvtColor(img, nv.COLOR_BGR2GRAY)
   hsv = nv.cvtColor(img, nv.COLOR_BGR2HSV)
   lab = nv.cvtColor(img, nv.COLOR_BGR2LAB)
   rgb = nv.cvtColor(img, nv.COLOR_BGR2RGB)
   
   # Split and merge channels
   b, g, r = nv.split(img)
   merged = nv.merge([r, g, b])  # RGB order
   
   # Adjust brightness/contrast
   bright = nv.convertScaleAbs(img, alpha=1.2, beta=30)

Contour Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to grayscale and threshold
   gray = nv.cvtColor(img, nv.COLOR_BGR2GRAY)
   _, binary = nv.threshold(gray, 127, 255, nv.THRESH_BINARY)
   
   # Find contours
   contours, hierarchy = nv.findContours(
       binary, nv.RETR_EXTERNAL, nv.CHAIN_APPROX_SIMPLE
   )
   
   # Analyze contours
   for cnt in contours:
       area = nv.contourArea(cnt)
       perimeter = nv.arcLength(cnt, True)
       
       if area > 100:  # Filter small contours
           # Bounding rectangle
           x, y, w, h = nv.boundingRect(cnt)
           nv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
           
           # Centroid
           M = nv.moments(cnt)
           if M["m00"] != 0:
               cx = int(M["m10"] / M["m00"])
               cy = int(M["m01"] / M["m00"])
               nv.circle(img, (cx, cy), 5, (0, 0, 255), -1)

Feature Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create ORB detector
   orb = nv.ORB_create(nFeatures=500)
   
   # Detect keypoints and compute descriptors
   kp, des = orb.detectAndCompute(gray, None)
   
   # Draw keypoints
   img_kp = nv.drawKeypoints(img, kp, None, 
                              color=(0, 255, 0),
                              flags=nv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

Histogram Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate histogram
   hist = nv.calcHist([gray], [0], None, [256], [0, 256])
   
   # Histogram equalization
   equalized = nv.equalizeHist(gray)
   
   # CLAHE (adaptive histogram equalization)
   clahe = nv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
   clahe_result = clahe.apply(gray)

Next Steps
----------

Now that you're familiar with the basics, explore these topics:

* :doc:`tutorials/index` - In-depth tutorials for common tasks
* :doc:`api/index` - Complete API reference
* :doc:`examples/index` - More code examples
* :doc:`performance_guide` - Optimization tips

Getting Help
------------

If you encounter issues:

1. Check the :doc:`faq` for common problems
2. Search `GitHub Issues <https://github.com/neurova/neurova/issues>`_
3. Ask on `Stack Overflow <https://stackoverflow.com/questions/tagged/neurova>`_ (tag: neurova)
4. Join our `Discord community <https://discord.gg/neurova>`_
