Image Processing Module
=======================

.. module:: neurova.imgproc
   :synopsis: Image processing functions

The imgproc module provides a comprehensive set of image processing
functions including filtering, geometric transformations, color
conversions, and feature detection.

Color Space Conversions
-----------------------

.. function:: cvtColor(src, code, dstCn=0)

   Convert an image from one color space to another.
   
   :param src: Input image
   :param code: Color conversion code (e.g., COLOR_BGR2GRAY)
   :param dstCn: Number of channels in the output image (0 = auto)
   :returns: Converted image
   :rtype: Mat

   Common conversion codes:
   
   * ``COLOR_BGR2GRAY`` - BGR to Grayscale
   * ``COLOR_GRAY2BGR`` - Grayscale to BGR
   * ``COLOR_BGR2RGB`` - BGR to RGB
   * ``COLOR_BGR2HSV`` - BGR to HSV
   * ``COLOR_HSV2BGR`` - HSV to BGR
   * ``COLOR_BGR2LAB`` - BGR to CIE L*a*b*
   * ``COLOR_LAB2BGR`` - CIE L*a*b* to BGR
   * ``COLOR_BGR2YCrCb`` - BGR to YCrCb
   * ``COLOR_YCrCb2BGR`` - YCrCb to BGR
   * ``COLOR_BGR2XYZ`` - BGR to CIE XYZ
   * ``COLOR_BGR2HLS`` - BGR to HLS

   Example:
   
   .. code-block:: python
   
      import neurova as nv
      
      img = nv.imread("color_image.jpg")
      gray = nv.cvtColor(img, nv.COLOR_BGR2GRAY)
      hsv = nv.cvtColor(img, nv.COLOR_BGR2HSV)

Image Filtering
---------------

Smoothing and Blurring
~~~~~~~~~~~~~~~~~~~~~~

.. function:: blur(src, ksize, anchor=(-1,-1), borderType=BORDER_DEFAULT)

   Blur an image using a normalized box filter.
   
   :param src: Input image
   :param ksize: Kernel size tuple (width, height)
   :param anchor: Anchor point (-1,-1 means center)
   :param borderType: Border extrapolation method
   :returns: Blurred image
   :rtype: Mat

.. function:: GaussianBlur(src, ksize, sigmaX, sigmaY=0, borderType=BORDER_DEFAULT)

   Blur an image using a Gaussian filter.
   
   :param src: Input image
   :param ksize: Kernel size (must be odd)
   :param sigmaX: Gaussian kernel standard deviation in X direction
   :param sigmaY: Gaussian kernel standard deviation in Y direction (0 = same as sigmaX)
   :param borderType: Border extrapolation method
   :returns: Blurred image
   :rtype: Mat

   Example:
   
   .. code-block:: python
   
      blurred = nv.GaussianBlur(img, (5, 5), 1.5)

.. function:: medianBlur(src, ksize)

   Blur an image using the median filter.
   
   :param src: Input image
   :param ksize: Aperture size (must be odd and > 1)
   :returns: Blurred image
   :rtype: Mat

.. function:: bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType=BORDER_DEFAULT)

   Apply bilateral filter (edge-preserving smoothing).
   
   :param src: Input 8-bit or floating-point image
   :param d: Diameter of pixel neighborhood (-1 for automatic)
   :param sigmaColor: Filter sigma in the color space
   :param sigmaSpace: Filter sigma in the coordinate space
   :returns: Filtered image
   :rtype: Mat

Edge Detection
~~~~~~~~~~~~~~

.. function:: Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)

   Find edges using the Canny algorithm.
   
   :param image: 8-bit input image
   :param threshold1: First threshold for hysteresis
   :param threshold2: Second threshold for hysteresis
   :param apertureSize: Aperture size for Sobel operator
   :param L2gradient: Use L2 norm for gradient magnitude
   :returns: Edge map
   :rtype: Mat

   Example:
   
   .. code-block:: python
   
      edges = nv.Canny(gray, 50, 150)

.. function:: Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0, borderType=BORDER_DEFAULT)

   Calculate image derivative using Sobel operator.
   
   :param src: Input image
   :param ddepth: Output image depth (-1 for same as source)
   :param dx: Order of derivative in X direction
   :param dy: Order of derivative in Y direction
   :param ksize: Kernel size (1, 3, 5, or 7)
   :param scale: Optional scale factor
   :param delta: Optional delta added to results
   :returns: Derivative image
   :rtype: Mat

.. function:: Laplacian(src, ddepth, ksize=1, scale=1, delta=0, borderType=BORDER_DEFAULT)

   Calculate the Laplacian of an image.

.. function:: Scharr(src, ddepth, dx, dy, scale=1, delta=0, borderType=BORDER_DEFAULT)

   Calculate image derivative using Scharr operator (more accurate than Sobel 3x3).

Custom Filtering
~~~~~~~~~~~~~~~~

.. function:: filter2D(src, ddepth, kernel, anchor=(-1,-1), delta=0, borderType=BORDER_DEFAULT)

   Convolve an image with a custom kernel.
   
   :param src: Input image
   :param ddepth: Output image depth
   :param kernel: Convolution kernel (single-channel)
   :param anchor: Anchor point
   :param delta: Value added to filtered results
   :returns: Filtered image
   :rtype: Mat

.. function:: sepFilter2D(src, ddepth, kernelX, kernelY, anchor=(-1,-1), delta=0, borderType=BORDER_DEFAULT)

   Apply separable linear filter.

.. function:: getGaussianKernel(ksize, sigma, ktype=float64)

   Returns Gaussian filter coefficients.
   
   :param ksize: Aperture size (must be odd)
   :param sigma: Gaussian standard deviation
   :param ktype: Type of filter coefficients
   :returns: 1D Gaussian kernel
   :rtype: Mat

Morphological Operations
------------------------

.. function:: erode(src, kernel, anchor=(-1,-1), iterations=1, borderType=BORDER_CONSTANT, borderValue=morphologyDefaultBorderValue())

   Erode an image using a structuring element.
   
   :param src: Input image
   :param kernel: Structuring element
   :param anchor: Anchor position
   :param iterations: Number of times to apply erosion
   :returns: Eroded image
   :rtype: Mat

.. function:: dilate(src, kernel, anchor=(-1,-1), iterations=1, borderType=BORDER_CONSTANT, borderValue=morphologyDefaultBorderValue())

   Dilate an image using a structuring element.

.. function:: morphologyEx(src, op, kernel, anchor=(-1,-1), iterations=1, borderType=BORDER_CONSTANT, borderValue=morphologyDefaultBorderValue())

   Perform advanced morphological transformations.
   
   :param src: Input image
   :param op: Type of morphological operation:
   
      * ``MORPH_OPEN`` - Opening (erode then dilate)
      * ``MORPH_CLOSE`` - Closing (dilate then erode)
      * ``MORPH_GRADIENT`` - Morphological gradient
      * ``MORPH_TOPHAT`` - Top hat
      * ``MORPH_BLACKHAT`` - Black hat
      * ``MORPH_HITMISS`` - Hit-or-miss transform
   
   :param kernel: Structuring element
   :returns: Processed image
   :rtype: Mat

.. function:: getStructuringElement(shape, ksize, anchor=(-1,-1))

   Create a structuring element of specified shape and size.
   
   :param shape: Element shape:
   
      * ``MORPH_RECT`` - Rectangle
      * ``MORPH_CROSS`` - Cross-shaped
      * ``MORPH_ELLIPSE`` - Ellipse
   
   :param ksize: Size of the structuring element
   :returns: Structuring element
   :rtype: Mat

Geometric Transformations
-------------------------

.. function:: resize(src, dsize, fx=0, fy=0, interpolation=INTER_LINEAR)

   Resize an image.
   
   :param src: Input image
   :param dsize: Output size (width, height) or (0, 0) to use fx/fy
   :param fx: Scale factor along horizontal axis
   :param fy: Scale factor along vertical axis
   :param interpolation: Interpolation method:
   
      * ``INTER_NEAREST`` - Nearest-neighbor
      * ``INTER_LINEAR`` - Bilinear interpolation
      * ``INTER_CUBIC`` - Bicubic interpolation
      * ``INTER_AREA`` - Pixel area relation (best for shrinking)
      * ``INTER_LANCZOS4`` - Lanczos interpolation
   
   :returns: Resized image
   :rtype: Mat

   Example:
   
   .. code-block:: python
   
      # Resize to specific size
      resized = nv.resize(img, (640, 480))
      
      # Resize by scale factor
      half = nv.resize(img, (0, 0), fx=0.5, fy=0.5)

.. function:: warpAffine(src, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=Scalar())

   Apply affine transformation to an image.
   
   :param src: Input image
   :param M: 2x3 transformation matrix
   :param dsize: Output image size
   :param flags: Interpolation method
   :param borderMode: Border extrapolation method
   :param borderValue: Border value (for BORDER_CONSTANT)
   :returns: Transformed image
   :rtype: Mat

.. function:: warpPerspective(src, M, dsize, flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, borderValue=Scalar())

   Apply perspective transformation.
   
   :param src: Input image
   :param M: 3x3 transformation matrix
   :param dsize: Output image size
   :returns: Transformed image
   :rtype: Mat

.. function:: getRotationMatrix2D(center, angle, scale)

   Calculate affine matrix for 2D rotation.
   
   :param center: Center of rotation
   :param angle: Rotation angle in degrees (counter-clockwise)
   :param scale: Isotropic scale factor
   :returns: 2x3 rotation matrix
   :rtype: Mat

.. function:: getAffineTransform(src, dst)

   Calculate affine transform from 3 pairs of corresponding points.
   
   :param src: Source triangle vertices (3 points)
   :param dst: Destination triangle vertices (3 points)
   :returns: 2x3 affine transformation matrix
   :rtype: Mat

.. function:: getPerspectiveTransform(src, dst)

   Calculate perspective transform from 4 pairs of corresponding points.
   
   :param src: Source quadrilateral vertices (4 points)
   :param dst: Destination quadrilateral vertices (4 points)
   :returns: 3x3 perspective transformation matrix
   :rtype: Mat

.. function:: remap(src, map1, map2, interpolation, borderMode=BORDER_CONSTANT, borderValue=Scalar())

   Apply generic geometric transformation.
   
   :param src: Input image
   :param map1: X coordinates or combined (x,y) map
   :param map2: Y coordinates (or empty if map1 is combined)
   :param interpolation: Interpolation method
   :returns: Remapped image
   :rtype: Mat

Thresholding
------------

.. function:: threshold(src, thresh, maxval, type)

   Apply fixed-level threshold to an image.
   
   :param src: Input array (single-channel)
   :param thresh: Threshold value
   :param maxval: Maximum value for THRESH_BINARY and THRESH_BINARY_INV
   :param type: Thresholding type:
   
      * ``THRESH_BINARY`` - Binary threshold
      * ``THRESH_BINARY_INV`` - Inverted binary
      * ``THRESH_TRUNC`` - Truncate
      * ``THRESH_TOZERO`` - Threshold to zero
      * ``THRESH_TOZERO_INV`` - Inverted threshold to zero
      * ``THRESH_OTSU`` - Otsu's algorithm (add to above)
      * ``THRESH_TRIANGLE`` - Triangle algorithm (add to above)
   
   :returns: Tuple (threshold_value, thresholded_image)
   :rtype: tuple

   Example:
   
   .. code-block:: python
   
      # Simple binary threshold
      _, binary = nv.threshold(gray, 127, 255, nv.THRESH_BINARY)
      
      # Otsu's automatic thresholding
      thresh, binary = nv.threshold(gray, 0, 255, 
                                     nv.THRESH_BINARY | nv.THRESH_OTSU)

.. function:: adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)

   Apply adaptive threshold using local pixel neighborhood.
   
   :param src: 8-bit single-channel image
   :param maxValue: Maximum value assigned to pixels exceeding threshold
   :param adaptiveMethod: ``ADAPTIVE_THRESH_MEAN_C`` or ``ADAPTIVE_THRESH_GAUSSIAN_C``
   :param thresholdType: ``THRESH_BINARY`` or ``THRESH_BINARY_INV``
   :param blockSize: Size of pixel neighborhood (must be odd)
   :param C: Constant subtracted from the mean
   :returns: Thresholded image
   :rtype: Mat

Histogram Operations
--------------------

.. function:: calcHist(images, channels, mask, histSize, ranges, accumulate=False)

   Calculate histogram of a set of arrays.
   
   :param images: Source images (list)
   :param channels: List of channel indices to analyze
   :param mask: Optional mask
   :param histSize: Array of histogram sizes for each dimension
   :param ranges: Array of ranges for each dimension
   :param accumulate: Accumulate the histogram
   :returns: Histogram
   :rtype: Mat

.. function:: equalizeHist(src)

   Equalize the histogram of a grayscale image.
   
   :param src: 8-bit single-channel image
   :returns: Equalized image
   :rtype: Mat

.. function:: compareHist(H1, H2, method)

   Compare two histograms.
   
   :param H1: First histogram
   :param H2: Second histogram
   :param method: Comparison method:
   
      * ``HISTCMP_CORREL`` - Correlation
      * ``HISTCMP_CHISQR`` - Chi-square
      * ``HISTCMP_INTERSECT`` - Intersection
      * ``HISTCMP_BHATTACHARYYA`` - Bhattacharyya distance
   
   :returns: Comparison result
   :rtype: float

.. function:: calcBackProject(images, channels, hist, ranges, scale=1)

   Calculate back projection of a histogram.

Contour Operations
------------------

.. function:: findContours(image, mode, method, offset=(0,0))

   Find contours in a binary image.
   
   :param image: 8-bit single-channel binary image
   :param mode: Contour retrieval mode:
   
      * ``RETR_EXTERNAL`` - Retrieve only external contours
      * ``RETR_LIST`` - Retrieve all contours as a flat list
      * ``RETR_CCOMP`` - Retrieve as 2-level hierarchy
      * ``RETR_TREE`` - Retrieve full hierarchy
   
   :param method: Contour approximation method:
   
      * ``CHAIN_APPROX_NONE`` - Store all points
      * ``CHAIN_APPROX_SIMPLE`` - Compress segments
      * ``CHAIN_APPROX_TC89_L1`` - Teh-Chin L1 approximation
      * ``CHAIN_APPROX_TC89_KCOS`` - Teh-Chin KCOS approximation
   
   :param offset: Optional offset for all contour points
   :returns: Tuple (contours, hierarchy)
   :rtype: tuple

   Example:
   
   .. code-block:: python
   
      contours, hierarchy = nv.findContours(binary, 
                                             nv.RETR_EXTERNAL,
                                             nv.CHAIN_APPROX_SIMPLE)
      
      for cnt in contours:
          area = nv.contourArea(cnt)
          print(f"Contour area: {area}")

.. function:: drawContours(image, contours, contourIdx, color, thickness=1, lineType=LINE_8, hierarchy=None, maxLevel=INT_MAX, offset=(0,0))

   Draw contours on an image.
   
   :param image: Destination image
   :param contours: List of contours
   :param contourIdx: Index of contour to draw (-1 for all)
   :param color: Contour color
   :param thickness: Line thickness (-1 to fill)
   :param lineType: Line connectivity type
   :returns: None (modifies image in place)

.. function:: contourArea(contour, oriented=False)

   Calculate contour area.
   
   :param contour: Input contour (vector of points)
   :param oriented: Return signed area if True
   :returns: Contour area
   :rtype: float

.. function:: arcLength(curve, closed)

   Calculate contour perimeter or curve length.
   
   :param curve: Input vector of 2D points
   :param closed: Whether the curve is closed
   :returns: Curve length
   :rtype: float

.. function:: boundingRect(points)

   Calculate bounding rectangle.
   
   :param points: Input 2D point set
   :returns: Bounding rectangle
   :rtype: Rect

.. function:: minAreaRect(points)

   Find minimum area rotated rectangle.
   
   :param points: Input 2D point set
   :returns: Rotated rectangle
   :rtype: RotatedRect

.. function:: minEnclosingCircle(points)

   Find minimum enclosing circle.
   
   :param points: Input 2D point set
   :returns: Tuple (center, radius)
   :rtype: tuple

.. function:: convexHull(points, clockwise=False, returnPoints=True)

   Find the convex hull of a point set.
   
   :param points: Input 2D point set
   :param clockwise: Orientation flag
   :param returnPoints: Return points (True) or indices (False)
   :returns: Convex hull points or indices
   :rtype: Mat

.. function:: approxPolyDP(curve, epsilon, closed)

   Approximate a polygonal curve with specified precision.
   
   :param curve: Input curve
   :param epsilon: Approximation accuracy (max distance)
   :param closed: Whether the curve is closed
   :returns: Approximated curve
   :rtype: Mat

Drawing Functions
-----------------

.. function:: line(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0)

   Draw a line segment.
   
   :param img: Image to draw on
   :param pt1: Start point
   :param pt2: End point
   :param color: Line color (Scalar or tuple)
   :param thickness: Line thickness
   :param lineType: Type of line (LINE_4, LINE_8, LINE_AA)

.. function:: rectangle(img, pt1, pt2, color, thickness=1, lineType=LINE_8, shift=0)

   Draw a rectangle.
   
   :param img: Image to draw on
   :param pt1: Top-left corner
   :param pt2: Bottom-right corner
   :param color: Rectangle color
   :param thickness: Line thickness (-1 to fill)

.. function:: circle(img, center, radius, color, thickness=1, lineType=LINE_8, shift=0)

   Draw a circle.
   
   :param img: Image to draw on
   :param center: Center point
   :param radius: Circle radius
   :param color: Circle color
   :param thickness: Line thickness (-1 to fill)

.. function:: ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=LINE_8, shift=0)

   Draw an ellipse or elliptic arc.
   
   :param img: Image to draw on
   :param center: Center of the ellipse
   :param axes: Half-axes (width/2, height/2)
   :param angle: Rotation angle in degrees
   :param startAngle: Starting angle of the arc
   :param endAngle: Ending angle of the arc
   :param color: Ellipse color
   :param thickness: Line thickness (-1 to fill)

.. function:: polylines(img, pts, isClosed, color, thickness=1, lineType=LINE_8, shift=0)

   Draw polygonal curves.
   
   :param img: Image to draw on
   :param pts: List of polygonal curves (list of point arrays)
   :param isClosed: Whether curves are closed
   :param color: Polyline color
   :param thickness: Line thickness

.. function:: fillPoly(img, pts, color, lineType=LINE_8, shift=0, offset=(0,0))

   Fill polygonal regions.

.. function:: putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=LINE_8, bottomLeftOrigin=False)

   Draw a text string.
   
   :param img: Image to draw on
   :param text: Text string
   :param org: Bottom-left corner of the text
   :param fontFace: Font type:
   
      * ``FONT_HERSHEY_SIMPLEX``
      * ``FONT_HERSHEY_PLAIN``
      * ``FONT_HERSHEY_DUPLEX``
      * ``FONT_HERSHEY_COMPLEX``
      * ``FONT_HERSHEY_TRIPLEX``
      * ``FONT_HERSHEY_COMPLEX_SMALL``
      * ``FONT_HERSHEY_SCRIPT_SIMPLEX``
      * ``FONT_HERSHEY_SCRIPT_COMPLEX``
   
   :param fontScale: Font scale factor
   :param color: Text color
   :param thickness: Line thickness

   Example:
   
   .. code-block:: python
   
      nv.putText(img, "Hello Neurova!", (50, 50),
                 nv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

.. function:: getTextSize(text, fontFace, fontScale, thickness)

   Calculate the width and height of a text string.
   
   :param text: Text string
   :param fontFace: Font type
   :param fontScale: Font scale factor
   :param thickness: Line thickness
   :returns: Tuple ((width, height), baseline)
   :rtype: tuple

Template Matching
-----------------

.. function:: matchTemplate(image, templ, method, mask=None)

   Compare a template against overlapped image regions.
   
   :param image: Image to search in
   :param templ: Template to search for
   :param method: Matching method:
   
      * ``TM_SQDIFF`` - Squared difference
      * ``TM_SQDIFF_NORMED`` - Normalized squared difference
      * ``TM_CCORR`` - Cross correlation
      * ``TM_CCORR_NORMED`` - Normalized cross correlation
      * ``TM_CCOEFF`` - Correlation coefficient
      * ``TM_CCOEFF_NORMED`` - Normalized correlation coefficient
   
   :param mask: Optional mask for the template
   :returns: Result map
   :rtype: Mat

   Example:
   
   .. code-block:: python
   
      result = nv.matchTemplate(img, template, nv.TM_CCOEFF_NORMED)
      min_val, max_val, min_loc, max_loc = nv.minMaxLoc(result)
      
      # For TM_CCOEFF_NORMED, max_loc is the best match
      top_left = max_loc
      bottom_right = (top_left[0] + template.shape[1],
                      top_left[1] + template.shape[0])
      nv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

See Also
--------

* :doc:`core` - Core array operations
* :doc:`imgcodecs` - Image file I/O
* :doc:`features2d` - Feature detection and description
