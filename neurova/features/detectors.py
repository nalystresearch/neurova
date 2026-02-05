# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Feature detection classes for Neurova.

Provides BRISK, MSER, FastFeatureDetector, GFTTDetector and SimpleBlobDetector.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


# Feature detector types
FAST_FEATURE_DETECTOR_TYPE_5_8 = 0
FAST_FEATURE_DETECTOR_TYPE_7_12 = 1
FAST_FEATURE_DETECTOR_TYPE_9_16 = 2


class KeyPoint:
    """Class representing a keypoint.
    
    Attributes:
        pt: Point coordinates (x, y)
        size: Diameter of meaningful keypoint neighborhood
        angle: Computed orientation of keypoint (-1 if not applicable)
        response: Response by which the most strong keypoints are selected
        octave: Octave (pyramid layer) from which keypoint was extracted
        class_id: Object class (if the keypoints need to be clustered)
    """
    
    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        size: float = 1,
        angle: float = -1,
        response: float = 0,
        octave: int = 0,
        class_id: int = -1
    ):
        self.pt = (x, y)
        self.size = size
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id
    
    def __repr__(self) -> str:
        return f"KeyPoint(pt={self.pt}, size={self.size})"


class FastFeatureDetector:
    """Wrapping class for feature detection using FAST method.
    
    Detects corners using the FAST algorithm.
    """
    
    def __init__(
        self,
        threshold: int = 10,
        nonmaxSuppression: bool = True,
        type: int = FAST_FEATURE_DETECTOR_TYPE_9_16
    ):
        """Initialize FAST detector.
        
        Args:
            threshold: Threshold on difference between intensity of center pixel
                and pixels of circle
            nonmaxSuppression: If true, non-maximum suppression is applied
            type: Neighborhood type (5_8, 7_12, 9_16)
        """
        self.threshold = threshold
        self.nonmaxSuppression = nonmaxSuppression
        self.type = type
    
    @staticmethod
    def create(
        threshold: int = 10,
        nonmaxSuppression: bool = True,
        type: int = FAST_FEATURE_DETECTOR_TYPE_9_16
    ) -> "FastFeatureDetector":
        """Create FastFeatureDetector."""
        return FastFeatureDetector(threshold, nonmaxSuppression, type)
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect keypoints in an image.
        
        Args:
            image: Image (grayscale)
            mask: Optional mask
        
        Returns:
            List of detected keypoints
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        keypoints = []
        
        # FAST-9 circle pattern (16 points)
        circle = [
            (0, 3), (1, 3), (2, 2), (3, 1),
            (3, 0), (3, -1), (2, -2), (1, -3),
            (0, -3), (-1, -3), (-2, -2), (-3, -1),
            (-3, 0), (-3, 1), (-2, 2), (-1, 3)
        ]
        
        n_contiguous = 9 if self.type == FAST_FEATURE_DETECTOR_TYPE_9_16 else 7
        
        responses = np.zeros((h, w), dtype=np.float32)
        
        for y in range(3, h - 3):
            for x in range(3, w - 3):
                if mask is not None and mask[y, x] == 0:
                    continue
                
                center = int(gray[y, x])
                thresh = self.threshold
                
                # Get circle pixels
                circle_vals = [int(gray[y + dy, x + dx]) for dx, dy in circle]
                
                # Check for corner (n contiguous brighter or darker)
                brighter = [v > center + thresh for v in circle_vals]
                darker = [v < center - thresh for v in circle_vals]
                
                # Check contiguous
                is_corner = False
                for start in range(16):
                    bright_count = sum(brighter[(start + i) % 16] for i in range(n_contiguous))
                    dark_count = sum(darker[(start + i) % 16] for i in range(n_contiguous))
                    
                    if bright_count == n_contiguous or dark_count == n_contiguous:
                        is_corner = True
                        break
                
                if is_corner:
                    # Compute response
                    response = sum(abs(v - center) for v in circle_vals if abs(v - center) > thresh)
                    responses[y, x] = response
                    
                    if not self.nonmaxSuppression:
                        keypoints.append(KeyPoint(x, y, 7, -1, response))
        
        # Non-maximum suppression
        if self.nonmaxSuppression:
            for y in range(3, h - 3):
                for x in range(3, w - 3):
                    if responses[y, x] > 0:
                        # Check 8-neighborhood
                        is_max = True
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                if responses[y + dy, x + dx] >= responses[y, x]:
                                    is_max = False
                                    break
                            if not is_max:
                                break
                        
                        if is_max:
                            keypoints.append(KeyPoint(x, y, 7, -1, responses[y, x]))
        
        return keypoints
    
    def getThreshold(self) -> int:
        return self.threshold
    
    def setThreshold(self, threshold: int) -> None:
        self.threshold = threshold
    
    def getNonmaxSuppression(self) -> bool:
        return self.nonmaxSuppression
    
    def setNonmaxSuppression(self, value: bool) -> None:
        self.nonmaxSuppression = value


class BRISK:
    """BRISK keypoint detector and descriptor extractor.
    
    Implements the BRISK (Binary Robust Invariant Scalable Keypoints) algorithm.
    """
    
    def __init__(
        self,
        thresh: int = 30,
        octaves: int = 3,
        patternScale: float = 1.0
    ):
        """Initialize BRISK.
        
        Args:
            thresh: AGAST detection threshold score
            octaves: Number of octaves for detection
            patternScale: Scale factor for pattern sampling points
        """
        self.thresh = thresh
        self.octaves = octaves
        self.patternScale = patternScale
        self._fast = FastFeatureDetector(thresh)
    
    @staticmethod
    def create(
        thresh: int = 30,
        octaves: int = 3,
        patternScale: float = 1.0
    ) -> "BRISK":
        """Create BRISK detector/descriptor."""
        return BRISK(thresh, octaves, patternScale)
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect keypoints in image.
        
        Args:
            image: Input image
            mask: Optional mask
        
        Returns:
            List of keypoints
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        all_keypoints = []
        
        for octave in range(self.octaves):
            scale = 2 ** octave
            scaled = gray[::scale, ::scale] if scale > 1 else gray
            
            kps = self._fast.detect(scaled, None)
            
            for kp in kps:
                x, y = kp.pt
                all_keypoints.append(KeyPoint(
                    x * scale, y * scale,
                    kp.size * scale,
                    kp.angle,
                    kp.response,
                    octave
                ))
        
        return all_keypoints
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[KeyPoint],
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Compute descriptors for keypoints.
        
        Args:
            image: Input image
            keypoints: Keypoints
            descriptors: Ignored
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        n_keypoints = len(keypoints)
        
        # BRISK uses 512 bits = 64 bytes per descriptor
        descriptors = np.zeros((n_keypoints, 64), dtype=np.uint8)
        
        # Simplified pattern - concentric circles
        n_points = 60
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radii = [2, 4, 6, 8, 10, 12]
        
        pattern = []
        for r in radii:
            for a in angles[:n_points // 6]:
                pattern.append((r * np.cos(a) * self.patternScale,
                               r * np.sin(a) * self.patternScale))
        
        valid_keypoints = []
        valid_descriptors = []
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Check if pattern fits in image
            if x < 15 or x >= w - 15 or y < 15 or y >= h - 15:
                continue
            
            # Sample pattern
            samples = []
            for px, py in pattern[:64]:
                sx, sy = int(x + px), int(y + py)
                if 0 <= sx < w and 0 <= sy < h:
                    samples.append(int(gray[sy, sx]))
                else:
                    samples.append(0)
            
            # Create binary descriptor from pairwise comparisons
            desc = np.zeros(64, dtype=np.uint8)
            for j in range(64):
                byte_val = 0
                for bit in range(8):
                    idx = j * 8 + bit
                    if idx + 1 < len(samples):
                        if samples[idx] > samples[(idx + 1) % len(samples)]:
                            byte_val |= (1 << bit)
                desc[j] = byte_val
            
            valid_keypoints.append(kp)
            valid_descriptors.append(desc)
        
        if valid_descriptors:
            return valid_keypoints, np.array(valid_descriptors, dtype=np.uint8)
        else:
            return [], np.zeros((0, 64), dtype=np.uint8)
    
    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors.
        
        Args:
            image: Input image
            mask: Optional mask
            descriptors: Ignored
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints = self.detect(image, mask)
        return self.compute(image, keypoints)


class MSER:
    """Maximally Stable Extremal Regions extractor.
    
    Detects MSER regions in images.
    """
    
    def __init__(
        self,
        delta: int = 5,
        minArea: int = 60,
        maxArea: int = 14400,
        maxVariation: float = 0.25,
        minDiversity: float = 0.2,
        maxEvolution: int = 200,
        areaThreshold: float = 1.01,
        minMargin: float = 0.003,
        edgeBlurSize: int = 5
    ):
        """Initialize MSER.
        
        Args:
            delta: Compares (size_{i}-size_{i-delta})/size_{i-delta}
            minArea: Minimum area of detected regions
            maxArea: Maximum area
            maxVariation: Maximum variation for stable regions
            minDiversity: Trace back parameter
            maxEvolution: Max evolution steps
            areaThreshold: Area threshold for reinitialization
            minMargin: Minimum margin
            edgeBlurSize: Edge blur size
        """
        self.delta = delta
        self.minArea = minArea
        self.maxArea = maxArea
        self.maxVariation = maxVariation
        self.minDiversity = minDiversity
        self.maxEvolution = maxEvolution
        self.areaThreshold = areaThreshold
        self.minMargin = minMargin
        self.edgeBlurSize = edgeBlurSize
    
    @staticmethod
    def create(
        delta: int = 5,
        minArea: int = 60,
        maxArea: int = 14400,
        maxVariation: float = 0.25,
        minDiversity: float = 0.2,
        maxEvolution: int = 200,
        areaThreshold: float = 1.01,
        minMargin: float = 0.003,
        edgeBlurSize: int = 5
    ) -> "MSER":
        """Create MSER extractor."""
        return MSER(delta, minArea, maxArea, maxVariation, minDiversity,
                   maxEvolution, areaThreshold, minMargin, edgeBlurSize)
    
    def detectRegions(
        self,
        image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[float, float, float, float, float]]]:
        """Detect MSER regions.
        
        Args:
            image: Input image (grayscale or BGR)
        
        Returns:
            Tuple of (list of point arrays, list of bounding boxes)
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        regions = []
        bboxes = []
        
        # Simplified MSER using threshold levels
        for thresh in range(0, 256, self.delta):
            binary = gray > thresh
            
            # Simple connected components
            from collections import deque
            
            visited = np.zeros_like(binary, dtype=bool)
            
            for y in range(h):
                for x in range(w):
                    if binary[y, x] and not visited[y, x]:
                        # BFS to find connected component
                        component = []
                        queue = deque([(x, y)])
                        visited[y, x] = True
                        
                        while queue and len(component) < self.maxArea:
                            cx, cy = queue.popleft()
                            component.append([cx, cy])
                            
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = cx + dx, cy + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    if binary[ny, nx] and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        queue.append((nx, ny))
                        
                        area = len(component)
                        if self.minArea <= area <= self.maxArea:
                            pts = np.array(component, dtype=np.int32)
                            regions.append(pts)
                            
                            x_min, y_min = pts.min(axis=0)
                            x_max, y_max = pts.max(axis=0)
                            cx = (x_min + x_max) / 2
                            cy = (y_min + y_max) / 2
                            w_box = x_max - x_min
                            h_box = y_max - y_min
                            bboxes.append((cx, cy, w_box, h_box, 0.0))
            
            if len(regions) >= 1000:  # Limit for performance
                break
        
        return regions, bboxes
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect keypoints from MSER regions.
        
        Args:
            image: Input image
            mask: Optional mask
        
        Returns:
            List of keypoints at region centers
        """
        regions, bboxes = self.detectRegions(image)
        
        keypoints = []
        for pts, bbox in zip(regions, bboxes):
            cx, cy, w, h, _ = bbox
            size = np.sqrt(w * h)
            keypoints.append(KeyPoint(cx, cy, size))
        
        return keypoints


class GFTTDetector:
    """Good Features to Track detector.
    
    Finds the most prominent corners using Shi-Tomasi or Harris method.
    """
    
    def __init__(
        self,
        maxCorners: int = 1000,
        qualityLevel: float = 0.01,
        minDistance: float = 1,
        blockSize: int = 3,
        useHarrisDetector: bool = False,
        k: float = 0.04
    ):
        """Initialize GFTT detector.
        
        Args:
            maxCorners: Maximum number of corners to return
            qualityLevel: Minimal accepted quality
            minDistance: Minimum distance between corners
            blockSize: Size of averaging block
            useHarrisDetector: Whether to use Harris detector
            k: Free parameter for Harris detector
        """
        self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance
        self.blockSize = blockSize
        self.useHarrisDetector = useHarrisDetector
        self.k = k
    
    @staticmethod
    def create(
        maxCorners: int = 1000,
        qualityLevel: float = 0.01,
        minDistance: float = 1,
        blockSize: int = 3,
        useHarrisDetector: bool = False,
        k: float = 0.04
    ) -> "GFTTDetector":
        """Create GFTT detector."""
        return GFTTDetector(maxCorners, qualityLevel, minDistance,
                           blockSize, useHarrisDetector, k)
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect corners in image.
        
        Args:
            image: Input image
            mask: Optional mask
        
        Returns:
            List of keypoints
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h, w = gray.shape
        
        # Compute gradients
        Ix = np.zeros_like(gray)
        Iy = np.zeros_like(gray)
        
        Ix[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        Iy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        # Compute structure tensor components
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Box filter
        bs = self.blockSize
        kernel = np.ones((bs, bs), dtype=np.float32) / (bs * bs)
        
        from scipy import ndimage
        Sxx = ndimage.convolve(Ixx, kernel)
        Syy = ndimage.convolve(Iyy, kernel)
        Sxy = ndimage.convolve(Ixy, kernel)
        
        # Compute corner response
        if self.useHarrisDetector:
            det = Sxx * Syy - Sxy * Sxy
            trace = Sxx + Syy
            response = det - self.k * trace * trace
        else:
            # Shi-Tomasi: minimum eigenvalue
            trace = Sxx + Syy
            det = Sxx * Syy - Sxy * Sxy
            discriminant = np.sqrt(np.maximum(trace * trace / 4 - det, 0))
            response = trace / 2 - discriminant
        
        # Apply mask
        if mask is not None:
            response = response * mask
        
        # Threshold
        threshold = self.qualityLevel * response.max()
        response[response < threshold] = 0
        
        # Non-maximum suppression and collect corners
        corners = []
        min_dist = int(self.minDistance)
        
        # Get all corner candidates
        candidates = []
        for y in range(min_dist, h - min_dist):
            for x in range(min_dist, w - min_dist):
                if response[y, x] > 0:
                    # Check if local maximum
                    is_max = True
                    for dy in range(-min_dist, min_dist + 1):
                        for dx in range(-min_dist, min_dist + 1):
                            if dy == 0 and dx == 0:
                                continue
                            if response[y + dy, x + dx] > response[y, x]:
                                is_max = False
                                break
                        if not is_max:
                            break
                    
                    if is_max:
                        candidates.append((response[y, x], x, y))
        
        # Sort by response and take top N
        candidates.sort(reverse=True)
        
        for resp, x, y in candidates[:self.maxCorners]:
            corners.append(KeyPoint(float(x), float(y), float(self.blockSize), -1, resp))
        
        return corners


class SimpleBlobDetector:
    """Simple blob detector based on thresholding and contour analysis."""
    
    class Params:
        """Parameters for SimpleBlobDetector."""
        
        def __init__(self):
            self.thresholdStep = 10
            self.minThreshold = 50
            self.maxThreshold = 220
            self.minRepeatability = 2
            self.minDistBetweenBlobs = 10
            
            self.filterByColor = True
            self.blobColor = 0
            
            self.filterByArea = True
            self.minArea = 25
            self.maxArea = 5000
            
            self.filterByCircularity = False
            self.minCircularity = 0.8
            self.maxCircularity = float('inf')
            
            self.filterByInertia = True
            self.minInertiaRatio = 0.1
            self.maxInertiaRatio = float('inf')
            
            self.filterByConvexity = True
            self.minConvexity = 0.95
            self.maxConvexity = float('inf')
    
    def __init__(self, params: Optional[Params] = None):
        """Initialize detector.
        
        Args:
            params: Detection parameters
        """
        self.params = params if params is not None else SimpleBlobDetector.Params()
    
    @staticmethod
    def create(params: Optional["SimpleBlobDetector.Params"] = None) -> "SimpleBlobDetector":
        """Create SimpleBlobDetector."""
        return SimpleBlobDetector(params)
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect blobs in image.
        
        Args:
            image: Input image
            mask: Optional mask
        
        Returns:
            List of keypoints at blob centers
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        blobs = []
        
        p = self.params
        
        # Iterate through thresholds
        for thresh in range(int(p.minThreshold), int(p.maxThreshold), int(p.thresholdStep)):
            if p.blobColor == 0:
                binary = gray < thresh
            else:
                binary = gray > thresh
            
            if mask is not None:
                binary = binary & (mask > 0)
            
            # Find connected components
            from collections import deque
            
            visited = np.zeros_like(binary, dtype=bool)
            
            for y in range(h):
                for x in range(w):
                    if binary[y, x] and not visited[y, x]:
                        component = []
                        queue = deque([(x, y)])
                        visited[y, x] = True
                        
                        while queue and len(component) < p.maxArea + 100:
                            cx, cy = queue.popleft()
                            component.append([cx, cy])
                            
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = cx + dx, cy + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    if binary[ny, nx] and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        queue.append((nx, ny))
                        
                        area = len(component)
                        
                        # Filter by area
                        if p.filterByArea:
                            if area < p.minArea or area > p.maxArea:
                                continue
                        
                        # Compute center
                        pts = np.array(component)
                        cx = np.mean(pts[:, 0])
                        cy = np.mean(pts[:, 1])
                        
                        # Compute radius
                        radius = np.sqrt(area / np.pi)
                        
                        blobs.append(KeyPoint(cx, cy, radius * 2, -1, float(area)))
        
        # Merge nearby blobs
        if len(blobs) > 0:
            merged = []
            used = [False] * len(blobs)
            
            for i, blob in enumerate(blobs):
                if used[i]:
                    continue
                
                cluster = [blob]
                used[i] = True
                
                for j in range(i + 1, len(blobs)):
                    if used[j]:
                        continue
                    
                    dist = np.sqrt((blob.pt[0] - blobs[j].pt[0])**2 + 
                                   (blob.pt[1] - blobs[j].pt[1])**2)
                    
                    if dist < p.minDistBetweenBlobs:
                        cluster.append(blobs[j])
                        used[j] = True
                
                if len(cluster) >= p.minRepeatability:
                    # Average cluster
                    avg_x = np.mean([b.pt[0] for b in cluster])
                    avg_y = np.mean([b.pt[1] for b in cluster])
                    avg_size = np.mean([b.size for b in cluster])
                    merged.append(KeyPoint(avg_x, avg_y, avg_size))
            
            return merged
        
        return blobs


def goodFeaturesToTrack(
    image: np.ndarray,
    maxCorners: int,
    qualityLevel: float,
    minDistance: float,
    mask: Optional[np.ndarray] = None,
    blockSize: int = 3,
    useHarrisDetector: bool = False,
    k: float = 0.04
) -> np.ndarray:
    """Find the strongest corners in an image.
    
    Args:
        image: Input 8-bit or floating-point 32-bit, single-channel image
        maxCorners: Maximum number of corners to return
        qualityLevel: Parameter characterizing the minimal accepted quality
        minDistance: Minimum possible Euclidean distance between corners
        mask: Optional region of interest
        blockSize: Size of averaging block for derivative covariance matrix
        useHarrisDetector: Use Harris detector
        k: Free parameter of Harris detector
    
    Returns:
        Output vector of detected corners (Nx1x2)
    """
    detector = GFTTDetector(
        maxCorners, qualityLevel, minDistance,
        blockSize, useHarrisDetector, k
    )
    
    keypoints = detector.detect(image, mask)
    
    if len(keypoints) == 0:
        return np.zeros((0, 1, 2), dtype=np.float32)
    
    corners = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    return corners.reshape(-1, 1, 2)


__all__ = [
    # Classes
    "KeyPoint",
    "FastFeatureDetector",
    "BRISK",
    "MSER",
    "GFTTDetector",
    "SimpleBlobDetector",
    # Functions
    "goodFeaturesToTrack",
    # Constants
    "FAST_FEATURE_DETECTOR_TYPE_5_8",
    "FAST_FEATURE_DETECTOR_TYPE_7_12",
    "FAST_FEATURE_DETECTOR_TYPE_9_16",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.