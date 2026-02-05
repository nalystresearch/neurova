Neurova Core Module
===================

.. module:: neurova.core
   :synopsis: Core image and array operations

The core module provides fundamental data structures and operations
for image processing and computer vision tasks.

Mat Class
---------

.. class:: Mat(rows=0, cols=0, dtype=uint8, channels=1)

   The primary n-dimensional dense array class.
   
   :param rows: Number of rows (height)
   :param cols: Number of columns (width)
   :param dtype: Data type (uint8, float32, etc.)
   :param channels: Number of channels (1, 3, 4)
   
   .. attribute:: shape
   
      Tuple of array dimensions (height, width, channels)
   
   .. attribute:: dtype
   
      Data type of the array elements
   
   .. attribute:: data
   
      Raw data buffer (memoryview)
   
   .. attribute:: size
   
      Total number of elements
   
   .. method:: clone()
   
      Create a deep copy of the array.
      
      :returns: New Mat with copied data
      :rtype: Mat
   
   .. method:: copyTo(dst)
   
      Copy the array to another Mat.
      
      :param dst: Destination Mat
   
   .. method:: convertTo(dtype, alpha=1.0, beta=0.0)
   
      Convert array to different type with optional scaling.
      
      :param dtype: Target data type
      :param alpha: Scale factor
      :param beta: Delta added to scaled values
      :returns: Converted array
      :rtype: Mat
   
   .. method:: reshape(channels, rows=0)
   
      Change the shape without copying data.
      
      :param channels: New number of channels
      :param rows: New number of rows (0 means unchanged)
      :returns: Reshaped array view
      :rtype: Mat
   
   .. method:: roi(rect)
   
      Extract a region of interest.
      
      :param rect: Tuple (x, y, width, height)
      :returns: Array view of the region
      :rtype: Mat

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   import neurova as nv
   
   # Create an empty image
   img = nv.Mat(480, 640, nv.uint8, 3)
   
   # Load from file
   img = nv.imread("image.jpg")
   
   # Access shape
   print(f"Shape: {img.shape}")  # (480, 640, 3)
   
   # Convert to grayscale
   gray = nv.cvtColor(img, nv.COLOR_BGR2GRAY)
   
   # Extract ROI
   roi = img.roi((100, 100, 200, 200))

Point Classes
-------------

.. class:: Point(x=0, y=0)

   2D point with integer coordinates.
   
   :param x: X coordinate
   :param y: Y coordinate

.. class:: Point2f(x=0.0, y=0.0)

   2D point with floating-point coordinates.

.. class:: Point3f(x=0.0, y=0.0, z=0.0)

   3D point with floating-point coordinates.

Size Classes
------------

.. class:: Size(width=0, height=0)

   2D size with integer dimensions.
   
   :param width: Width
   :param height: Height
   
   .. attribute:: area
   
      Computed area (width * height)

Rect Class
----------

.. class:: Rect(x=0, y=0, width=0, height=0)

   2D rectangle with integer coordinates.
   
   :param x: Top-left x coordinate
   :param y: Top-left y coordinate
   :param width: Rectangle width
   :param height: Rectangle height
   
   .. method:: contains(point)
   
      Check if a point is inside the rectangle.
      
      :param point: Point to check
      :returns: True if point is inside
      :rtype: bool
   
   .. method:: area()
   
      Compute the rectangle area.
      
      :returns: Area (width * height)
      :rtype: int

Scalar Class
------------

.. class:: Scalar(v0=0, v1=0, v2=0, v3=0)

   4-element vector, typically used for pixel values.
   
   :param v0: First channel value
   :param v1: Second channel value
   :param v2: Third channel value
   :param v3: Fourth channel value

Array Operations
----------------

.. function:: zeros(rows, cols, dtype=uint8, channels=1)

   Create an array filled with zeros.
   
   :param rows: Number of rows
   :param cols: Number of columns
   :param dtype: Data type
   :param channels: Number of channels
   :returns: Zero-initialized array
   :rtype: Mat

.. function:: ones(rows, cols, dtype=uint8, channels=1)

   Create an array filled with ones.

.. function:: eye(size, dtype=float32)

   Create an identity matrix.
   
   :param size: Matrix size (square)
   :param dtype: Data type
   :returns: Identity matrix
   :rtype: Mat

.. function:: add(src1, src2, dst=None, mask=None, dtype=-1)

   Element-wise array addition.
   
   :param src1: First input array
   :param src2: Second input array
   :param dst: Output array (optional)
   :param mask: Operation mask (optional)
   :param dtype: Output data type (-1 for same as input)
   :returns: Result array
   :rtype: Mat

.. function:: subtract(src1, src2, dst=None, mask=None, dtype=-1)

   Element-wise array subtraction.

.. function:: multiply(src1, src2, dst=None, scale=1.0, dtype=-1)

   Element-wise array multiplication.

.. function:: divide(src1, src2, dst=None, scale=1.0, dtype=-1)

   Element-wise array division.

.. function:: absdiff(src1, src2, dst=None)

   Absolute difference of two arrays.

.. function:: mean(src, mask=None)

   Calculate the mean of array elements.
   
   :param src: Input array
   :param mask: Optional mask
   :returns: Mean values per channel
   :rtype: Scalar

.. function:: meanStdDev(src, mean, stddev, mask=None)

   Calculate mean and standard deviation.

.. function:: minMaxLoc(src, mask=None)

   Find minimum and maximum element values and locations.
   
   :param src: Input single-channel array
   :param mask: Optional mask
   :returns: Tuple (minVal, maxVal, minLoc, maxLoc)
   :rtype: tuple

.. function:: normalize(src, dst, alpha=1, beta=0, norm_type=NORM_L2, dtype=-1, mask=None)

   Normalize array values.
   
   :param src: Input array
   :param dst: Output array
   :param alpha: Norm value or lower range bound
   :param beta: Upper range bound (for NORM_MINMAX)
   :param norm_type: Normalization type
   :param dtype: Output data type
   :param mask: Optional mask

Matrix Operations
-----------------

.. function:: transpose(src, dst=None)

   Transpose a matrix.

.. function:: flip(src, flipCode)

   Flip array in one of three ways.
   
   :param src: Input array
   :param flipCode: 0 for vertical, 1 for horizontal, -1 for both
   :returns: Flipped array
   :rtype: Mat

.. function:: rotate(src, rotateCode)

   Rotate array by 90 degrees.
   
   :param src: Input array
   :param rotateCode: ROTATE_90_CLOCKWISE, ROTATE_180, or ROTATE_90_COUNTERCLOCKWISE
   :returns: Rotated array
   :rtype: Mat

.. function:: vconcat(src1, src2, dst=None)

   Vertical concatenation of arrays.

.. function:: hconcat(src1, src2, dst=None)

   Horizontal concatenation of arrays.

Channel Operations
------------------

.. function:: split(src)

   Split multi-channel array into separate channels.
   
   :param src: Multi-channel input array
   :returns: List of single-channel arrays
   :rtype: list[Mat]

.. function:: merge(mv)

   Merge separate channels into a multi-channel array.
   
   :param mv: List of single-channel arrays
   :returns: Multi-channel array
   :rtype: Mat

.. function:: mixChannels(src, dst, fromTo)

   Copy specified channels from input to output.
   
   :param src: Input arrays
   :param dst: Output arrays
   :param fromTo: Channel mapping pairs

See Also
--------

* :doc:`imgproc` - Image processing functions
* :doc:`imgcodecs` - Image file I/O
* :doc:`highgui` - GUI and display functions
