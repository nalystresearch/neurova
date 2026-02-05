Deep Neural Network Module
===========================

.. module:: neurova.dnn
   :synopsis: Deep neural network inference

The dnn module provides functionality for neural network inference,
supporting multiple model formats and hardware acceleration backends.

Model Loading
-------------

.. function:: readNet(model, config="", framework="")

   Read a deep learning network from a model file.
   
   :param model: Path to the binary model file
   :param config: Path to text network configuration file (optional)
   :param framework: Explicit framework name ("caffe", "neurova_pb", "neurova_native", "darknet", "onnx")
   :returns: Loaded network
   :rtype: Net
   
   Example:
   
   .. code-block:: python
   
      import neurova as nv
      
      # Load ONNX model
      net = nv.dnn.readNet("model.onnx")
      
      # Load Neurova model
      net = nv.dnn.readNet("frozen_graph.pb")
      
      # Load Caffe model
      net = nv.dnn.readNet("model.caffemodel", "deploy.prototxt")

.. function:: readNetFromONNX(onnxFile)

   Read network from ONNX model file.
   
   :param onnxFile: Path to .onnx file
   :returns: Loaded network
   :rtype: Net

.. function:: readNetFromNeurova(model, config="")

   Read network from Neurova model.
   
   :param model: Path to .pb file
   :param config: Path to .pbtxt graph definition (optional)
   :returns: Loaded network
   :rtype: Net

.. function:: readNetFromCaffe(prototxt, caffeModel="")

   Read network from Caffe model.
   
   :param prototxt: Path to .prototxt file
   :param caffeModel: Path to .caffemodel file (optional)
   :returns: Loaded network
   :rtype: Net

.. function:: readNetFromDarknet(cfgFile, darknetModel="")

   Read network from Darknet configuration and weights.
   
   :param cfgFile: Path to .cfg file
   :param darknetModel: Path to .weights file (optional)
   :returns: Loaded network
   :rtype: Net

.. function:: readNetFromNeurovaModel(model, isBinary=True)

   Read network from Neurova native model.
   
   :param model: Path to neurova .nvm/.net file
   :param isBinary: Whether the model is binary
   :returns: Loaded network
   :rtype: Net

.. function:: readNetFromModelOptimizer(xml, bin="")

   Read network from OpenVINO IR format.
   
   :param xml: Path to .xml model description
   :param bin: Path to .bin weights file
   :returns: Loaded network
   :rtype: Net

Net Class
---------

.. class:: Net

   Neural network inference engine.
   
   .. method:: setInput(blob, name="", scalefactor=1.0, mean=Scalar())
   
      Set the input blob for the network.
      
      :param blob: Input blob (4D: NCHW format)
      :param name: Name of the input layer
      :param scalefactor: Multiplier for input values
      :param mean: Scalar to subtract from input
   
   .. method:: forward(outputName="")
   
      Run forward pass and return output of specified layer.
      
      :param outputName: Name of output layer (empty = default output)
      :returns: Output blob
      :rtype: Mat
      
      Example:
      
      .. code-block:: python
      
         net.setInput(blob)
         output = net.forward()
   
   .. method:: forward(outputNames)
   
      Run forward pass and return outputs of multiple layers.
      
      :param outputNames: List of output layer names
      :returns: List of output blobs
      :rtype: list[Mat]
   
   .. method:: setPreferableBackend(backendId)
   
      Set the preferred DNN backend.
      
      :param backendId: Backend identifier:
      
         * ``DNN_BACKEND_DEFAULT`` - Auto-select
         * ``DNN_BACKEND_NEUROVA`` - Neurova CPU backend
         * ``DNN_BACKEND_CUDA`` - NVIDIA CUDA backend
         * ``DNN_BACKEND_VKCOM`` - Vulkan Compute backend
         * ``DNN_BACKEND_INFERENCE_ENGINE`` - Intel OpenVINO
         * ``DNN_BACKEND_COREML`` - Apple Core ML
         * ``DNN_BACKEND_NNAPI`` - Android NNAPI
   
   .. method:: setPreferableTarget(targetId)
   
      Set the preferred computation target.
      
      :param targetId: Target identifier:
      
         * ``DNN_TARGET_CPU`` - CPU
         * ``DNN_TARGET_OPENCL`` - OpenCL
         * ``DNN_TARGET_OPENCL_FP16`` - OpenCL half-precision
         * ``DNN_TARGET_CUDA`` - CUDA
         * ``DNN_TARGET_CUDA_FP16`` - CUDA half-precision
         * ``DNN_TARGET_VULKAN`` - Vulkan
         * ``DNN_TARGET_MYRIAD`` - Intel Movidius
         * ``DNN_TARGET_FPGA`` - Intel FPGA
   
   .. method:: getLayerNames()
   
      Get names of all network layers.
      
      :returns: List of layer names
      :rtype: list[str]
   
   .. method:: getUnconnectedOutLayers()
   
      Get indices of output layers (layers without consumers).
      
      :returns: List of layer indices
      :rtype: list[int]
   
   .. method:: getUnconnectedOutLayersNames()
   
      Get names of output layers.
      
      :returns: List of output layer names
      :rtype: list[str]
   
   .. method:: getLayer(layerId)
   
      Get layer by ID or name.
      
      :param layerId: Layer index or name
      :returns: Layer object
      :rtype: Layer
   
   .. method:: enableFusion(enable)
   
      Enable or disable layer fusion optimization.
      
      :param enable: Whether to enable fusion
   
   .. method:: enableWinograd(enable)
   
      Enable or disable Winograd convolution optimization.
      
      :param enable: Whether to enable Winograd
   
   .. method:: getPerfProfile()
   
      Get performance timing for layers.
      
      :returns: Tuple (total_time, layer_times)
      :rtype: tuple
   
   .. method:: getFLOPS(netInputShapes)
   
      Calculate FLOPS for the network.
      
      :param netInputShapes: List of input shapes
      :returns: Number of floating-point operations
      :rtype: int
   
   .. method:: getMemoryConsumption(netInputShapes)
   
      Estimate memory consumption.
      
      :param netInputShapes: List of input shapes
      :returns: Tuple (weights_size, blobs_size)
      :rtype: tuple

Blob Functions
--------------

.. function:: blobFromImage(image, scalefactor=1.0, size=(0,0), mean=Scalar(), swapRB=False, crop=False, ddepth=CV_32F)

   Create a 4D blob from an image.
   
   :param image: Input image (BGR format)
   :param scalefactor: Multiplier for image values
   :param size: Spatial size for output blob
   :param mean: Scalar values to subtract from channels
   :param swapRB: Swap first and last channels (BGR to RGB)
   :param crop: Whether to center crop after resizing
   :param ddepth: Output blob depth (CV_32F or CV_8U)
   :returns: 4D blob (NCHW format)
   :rtype: Mat
   
   Example:
   
   .. code-block:: python
   
      # Prepare image for ImageNet-style network
      blob = nv.dnn.blobFromImage(img, 1.0/255.0, (224, 224),
                                   (0.485, 0.456, 0.406), 
                                   swapRB=True, crop=False)
      
      # For networks expecting [0, 255] range
      blob = nv.dnn.blobFromImage(img, 1.0, (300, 300),
                                   (104.0, 177.0, 123.0))

.. function:: blobFromImages(images, scalefactor=1.0, size=(0,0), mean=Scalar(), swapRB=False, crop=False, ddepth=CV_32F)

   Create a 4D blob from multiple images.
   
   :param images: List of input images
   :returns: 4D blob with batch dimension
   :rtype: Mat

.. function:: imagesFromBlob(blob)

   Extract images from a 4D blob.
   
   :param blob: 4D blob (NCHW format)
   :returns: List of images
   :rtype: list[Mat]

Detection and Recognition
-------------------------

.. function:: NMSBoxes(bboxes, scores, score_threshold, nms_threshold, eta=1.0, top_k=0)

   Perform non-maximum suppression on bounding boxes.
   
   :param bboxes: List of bounding boxes (x, y, w, h)
   :param scores: Corresponding confidence scores
   :param score_threshold: Minimum score to consider
   :param nms_threshold: NMS overlap threshold (IoU)
   :param eta: Coefficient for adaptive threshold
   :param top_k: Maximum number of boxes to keep (0 = all)
   :returns: Indices of selected boxes
   :rtype: list[int]
   
   Example:
   
   .. code-block:: python
   
      # After running detection network
      boxes = []
      confidences = []
      class_ids = []
      
      for detection in detections:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          
          if confidence > 0.5:
              box = detection[0:4] * np.array([W, H, W, H])
              boxes.append(box)
              confidences.append(float(confidence))
              class_ids.append(class_id)
      
      # Apply NMS
      indices = nv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

.. function:: softNMSBoxes(bboxes, scores, score_threshold, nms_threshold, updated_scores, sigma=0.5, top_k=0, method=SOFTNMS_GAUSSIAN)

   Perform soft non-maximum suppression.
   
   :param bboxes: List of bounding boxes
   :param scores: Corresponding confidence scores
   :param score_threshold: Minimum score
   :param nms_threshold: NMS overlap threshold
   :param updated_scores: Output updated scores
   :param sigma: Gaussian sigma (for SOFTNMS_GAUSSIAN)
   :param top_k: Maximum boxes to keep
   :param method: SOFTNMS_LINEAR or SOFTNMS_GAUSSIAN
   :returns: Indices of selected boxes
   :rtype: list[int]

.. function:: NMSBoxesBatched(bboxes, scores, class_ids, score_threshold, nms_threshold, eta=1.0, top_k=0)

   Perform batched NMS (separate NMS per class).
   
   :param bboxes: List of bounding boxes
   :param scores: Corresponding confidence scores
   :param class_ids: Class ID for each box
   :param score_threshold: Minimum score
   :param nms_threshold: NMS overlap threshold
   :returns: Indices of selected boxes
   :rtype: list[int]

Pre-trained Models
------------------

Classification
~~~~~~~~~~~~~~

.. class:: ClassificationModel(model, config="")

   High-level API for image classification.
   
   :param model: Path to model file
   :param config: Path to configuration file
   
   .. method:: classify(frame)
   
      Classify the input image.
      
      :param frame: Input image
      :returns: Tuple (classId, confidence)
      :rtype: tuple
   
   Example:
   
   .. code-block:: python
   
      classifier = nv.dnn.ClassificationModel("mobilenet_v2.onnx")
      classifier.setInputParams(scale=1/255.0, size=(224, 224), 
                                 mean=(0.485, 0.456, 0.406), swapRB=True)
      
      class_id, confidence = classifier.classify(image)
      print(f"Class: {class_id}, Confidence: {confidence:.2f}")

Object Detection
~~~~~~~~~~~~~~~~

.. class:: DetectionModel(model, config="")

   High-level API for object detection.
   
   :param model: Path to model file
   :param config: Path to configuration file
   
   .. method:: detect(frame, confThreshold=0.5, nmsThreshold=0.4)
   
      Detect objects in the input image.
      
      :param frame: Input image
      :param confThreshold: Confidence threshold
      :param nmsThreshold: NMS threshold
      :returns: Tuple (classIds, confidences, boxes)
      :rtype: tuple
   
   Example:
   
   .. code-block:: python
   
      detector = nv.dnn.DetectionModel("nvdetect_nano.onnx")
      detector.setInputParams(size=(640, 640), scale=1/255.0, swapRB=True)
      
      classes, scores, boxes = detector.detect(image)
      
      for classId, score, box in zip(classes, scores, boxes):
          nv.rectangle(image, box, (0, 255, 0), 2)
          label = f"{class_names[classId]}: {score:.2f}"
          nv.putText(image, label, (box[0], box[1] - 10),
                     nv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

Segmentation
~~~~~~~~~~~~

.. class:: SegmentationModel(model, config="")

   High-level API for semantic segmentation.
   
   :param model: Path to model file
   :param config: Path to configuration file
   
   .. method:: segment(frame)
   
      Perform semantic segmentation.
      
      :param frame: Input image
      :returns: Segmentation mask
      :rtype: Mat

Keypoint Detection
~~~~~~~~~~~~~~~~~~

.. class:: KeypointsModel(model, config="")

   High-level API for keypoint detection.
   
   :param model: Path to model file
   :param config: Path to configuration file
   
   .. method:: estimate(frame, thresh=0.5)
   
      Estimate keypoints in the input image.
      
      :param frame: Input image
      :param thresh: Confidence threshold
      :returns: List of keypoint locations
      :rtype: list

Face Detection
~~~~~~~~~~~~~~

.. class:: FaceDetectorYN

   YuNet face detection model.
   
   .. classmethod:: create(model, config, input_size, score_threshold=0.9, nms_threshold=0.3, top_k=5000, backend_id=0, target_id=0)
   
      Create a FaceDetectorYN instance.
      
      :param model: Path to model file
      :param config: Path to config file (empty for ONNX)
      :param input_size: Input size tuple (width, height)
      :param score_threshold: Score threshold
      :param nms_threshold: NMS threshold
      :param top_k: Maximum detections to keep
      :returns: FaceDetectorYN instance
   
   .. method:: detect(image)
   
      Detect faces in the input image.
      
      :param image: Input image
      :returns: Detection results (N x 15 matrix)
      :rtype: Mat
      
      Each row contains:
      [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
      
      Where: re=right eye, le=left eye, nt=nose tip, rcm=right corner of mouth, lcm=left corner of mouth
   
   Example:
   
   .. code-block:: python
   
      detector = nv.dnn.FaceDetectorYN.create(
          "yunet.onnx", "", (320, 320),
          score_threshold=0.9, nms_threshold=0.3
      )
      detector.setInputSize((image.shape[1], image.shape[0]))
      
      _, faces = detector.detect(image)
      
      for face in faces:
          x, y, w, h = face[:4].astype(int)
          nv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

Face Recognition
~~~~~~~~~~~~~~~~

.. class:: FaceRecognizerSF

   SFace face recognition model.
   
   .. classmethod:: create(model, config, backend_id=0, target_id=0)
   
      Create a FaceRecognizerSF instance.
      
      :param model: Path to model file
      :param config: Path to config file
      :returns: FaceRecognizerSF instance
   
   .. method:: alignCrop(src_img, face_box)
   
      Align and crop a face from the image.
      
      :param src_img: Source image
      :param face_box: Face detection result
      :returns: Aligned face image
      :rtype: Mat
   
   .. method:: feature(aligned_img)
   
      Extract face feature embedding.
      
      :param aligned_img: Aligned face image
      :returns: 128-dimensional feature vector
      :rtype: Mat
   
   .. method:: match(face_feature1, face_feature2, dis_type=FR_COSINE)
   
      Match two face features.
      
      :param face_feature1: First face feature
      :param face_feature2: Second face feature
      :param dis_type: Distance type (FR_COSINE or FR_NORM_L2)
      :returns: Match score
      :rtype: float

Text Detection and Recognition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. class:: TextDetectionModel_DB

   Differentiable Binarization text detector.
   
   .. method:: detect(frame)
   
      Detect text regions in the input image.
      
      :param frame: Input image
      :returns: List of detected text contours
      :rtype: list

.. class:: TextRecognitionModel

   CRNN-based text recognition.
   
   .. method:: recognize(frame)
   
      Recognize text in the input image.
      
      :param frame: Input cropped text image
      :returns: Recognized text
      :rtype: str

Backend Configuration
---------------------

.. function:: getAvailableBackends()

   Get list of available DNN backends.
   
   :returns: List of (backend_id, target_id) tuples
   :rtype: list[tuple]

.. function:: getAvailableTargets(backend)

   Get available targets for a backend.
   
   :param backend: Backend ID
   :returns: List of target IDs
   :rtype: list[int]

Example Workflows
-----------------

Complete Object Detection Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   import numpy as np
   
   # Load COCO class names
   with open("coco.names", "r") as f:
       classes = f.read().strip().split("\n")
   
   # Load NVDetect model
   net = nv.dnn.readNet("nvdetect_nano.onnx")
   net.setPreferableBackend(nv.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(nv.dnn.DNN_TARGET_CUDA)
   
   # Read and preprocess image
   image = nv.imread("input.jpg")
   blob = nv.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True)
   
   # Run inference
   net.setInput(blob)
   outputs = net.forward(net.getUnconnectedOutLayersNames())
   
   # Post-process results
   detections = outputs[0][0]
   height, width = image.shape[:2]
   
   boxes, confidences, class_ids = [], [], []
   
   for detection in detections:
       scores = detection[4:]
       class_id = np.argmax(scores)
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
       label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
       nv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
       nv.putText(image, label, (x, y-10),
                  nv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
   nv.imwrite("output.jpg", image)

Real-time Face Detection
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Initialize face detector
   detector = nv.dnn.FaceDetectorYN.create(
       "yunet.onnx", "", (320, 320),
       score_threshold=0.9, nms_threshold=0.3
   )
   
   # Open camera
   cap = nv.VideoCapture(0)
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       # Set input size to frame size
       detector.setInputSize((frame.shape[1], frame.shape[0]))
       
       # Detect faces
       _, faces = detector.detect(frame)
       
       if faces is not None:
           for face in faces:
               box = face[:4].astype(int)
               landmarks = face[4:14].reshape(5, 2).astype(int)
               
               # Draw bounding box
               nv.rectangle(frame, 
                           (box[0], box[1]),
                           (box[0]+box[2], box[1]+box[3]),
                           (0, 255, 0), 2)
               
               # Draw landmarks
               for lm in landmarks:
                   nv.circle(frame, tuple(lm), 2, (255, 0, 0), -1)
       
       nv.imshow("Face Detection", frame)
       if nv.waitKey(1) & 0xFF == ord('q'):
           break
   
   cap.release()
   nv.destroyAllWindows()

See Also
--------

* :doc:`core` - Core array operations
* :doc:`imgproc` - Image preprocessing functions
* :doc:`videoio` - Video capture for real-time inference
