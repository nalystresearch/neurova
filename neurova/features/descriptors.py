# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.features.descriptors - Feature Detectors and Descriptors

Provides feature detection and description algorithms.
"""

from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np


class KeyPoint:
    """Represents a detected keypoint."""
    
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        size: float = 0.0,
        angle: float = -1.0,
        response: float = 0.0,
        octave: int = 0,
        class_id: int = -1
    ):
        """Initialize keypoint.
        
        Args:
            x: X coordinate
            y: Y coordinate
            size: Diameter of meaningful keypoint neighborhood
            angle: Orientation in degrees (0-360, -1 if not applicable)
            response: Response (strength) of keypoint
            octave: Octave (pyramid layer) from which keypoint was extracted
            class_id: Object class (for matching)
        """
        self.pt = (float(x), float(y))
        self.size = float(size)
        self.angle = float(angle)
        self.response = float(response)
        self.octave = int(octave)
        self.class_id = int(class_id)
    
    def __repr__(self):
        return f"KeyPoint(pt={self.pt}, size={self.size}, angle={self.angle})"


class Feature2D:
    """Base class for 2D feature detectors and descriptors."""
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect keypoints in an image."""
        raise NotImplementedError
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[KeyPoint],
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Compute descriptors for keypoints."""
        raise NotImplementedError
    
    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors."""
        keypoints = self.detect(image, mask)
        return self.compute(image, keypoints, descriptors)


class ORB(Feature2D):
    """Oriented FAST and Rotated BRIEF feature detector and descriptor."""
    
    def __init__(
        self,
        nfeatures: int = 500,
        scaleFactor: float = 1.2,
        nlevels: int = 8,
        edgeThreshold: int = 31,
        firstLevel: int = 0,
        WTA_K: int = 2,
        scoreType: int = 0,  # HARRIS_SCORE
        patchSize: int = 31,
        fastThreshold: int = 20
    ):
        """Initialize ORB detector.
        
        Args:
            nfeatures: Maximum number of features to retain
            scaleFactor: Pyramid decimation ratio
            nlevels: Number of pyramid levels
            edgeThreshold: Border size where features are not detected
            firstLevel: Level of pyramid to put source image
            WTA_K: Number of points for orientation (2, 3, or 4)
            scoreType: HARRIS_SCORE=0 or FAST_SCORE=1
            patchSize: Size of patch used for descriptor
            fastThreshold: FAST threshold
        """
        self._nfeatures = nfeatures
        self._scaleFactor = scaleFactor
        self._nlevels = nlevels
        self._edgeThreshold = edgeThreshold
        self._firstLevel = firstLevel
        self._WTA_K = WTA_K
        self._scoreType = scoreType
        self._patchSize = patchSize
        self._fastThreshold = fastThreshold
        
        # Pre-compute BRIEF pattern
        self._pattern = self._generate_brief_pattern()
    
    def _generate_brief_pattern(self) -> np.ndarray:
        """Generate sampling pattern for BRIEF descriptor."""
        np.random.seed(42)  # For reproducibility
        pattern = np.random.randint(
            -self._patchSize // 2, 
            self._patchSize // 2 + 1,
            size=(256, 4)  # 256 bits, 2 points each
        )
        return pattern
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect ORB keypoints.
        
        Args:
            image: Input image (grayscale or color)
            mask: Optional mask
        
        Returns:
            List of detected keypoints
        """
        if image.ndim == 3:
            from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Build pyramid
        keypoints = []
        current = gray.astype(np.float32)
        
        for level in range(self._nlevels):
            scale = self._scaleFactor ** level
            
            # FAST corner detection
            corners = self._fast_detect(current, self._fastThreshold)
            
            # Apply mask if provided
            if mask is not None:
                scaled_mask = mask
                if level > 0:
                    from scipy.ndimage import zoom
                    scaled_mask = zoom(mask, 1.0 / scale, order=0) > 0
                
                corners = [(x, y, r) for x, y, r in corners 
                          if scaled_mask[int(y), int(x)]]
            
            # Compute Harris response for ranking
            if self._scoreType == 0:  # HARRIS_SCORE
                corners = self._compute_harris_response(current, corners)
            
            # Filter by edge threshold
            h, w = current.shape
            et = self._edgeThreshold
            corners = [(x, y, r) for x, y, r in corners
                      if et <= x < w - et and et <= y < h - et]
            
            # Convert to keypoints
            for x, y, response in corners:
                kp = KeyPoint(
                    x=x * scale,
                    y=y * scale,
                    size=self._patchSize * scale,
                    angle=-1,  # Will be computed later
                    response=response,
                    octave=level
                )
                keypoints.append(kp)
            
            # Downsample for next level
            if level < self._nlevels - 1:
                new_size = (int(current.shape[1] / self._scaleFactor),
                           int(current.shape[0] / self._scaleFactor))
                if new_size[0] < self._patchSize or new_size[1] < self._patchSize:
                    break
                from scipy.ndimage import zoom
                current = zoom(current, 1.0 / self._scaleFactor, order=1)
        
        # Sort by response and keep top N
        keypoints.sort(key=lambda kp: kp.response, reverse=True)
        keypoints = keypoints[:self._nfeatures]
        
        # Compute orientations
        if image.ndim == 3:
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image
        
        for kp in keypoints:
            kp.angle = self._compute_orientation(gray, kp)
        
        return keypoints
    
    def _fast_detect(
        self,
        image: np.ndarray,
        threshold: int
    ) -> List[Tuple[int, int, float]]:
        """Simple FAST corner detection."""
        h, w = image.shape
        corners = []
        
        # FAST-9 pattern (simplified)
        circle = [
            (0, -3), (1, -3), (2, -2), (3, -1),
            (3, 0), (3, 1), (2, 2), (1, 3),
            (0, 3), (-1, 3), (-2, 2), (-3, 1),
            (-3, 0), (-3, -1), (-2, -2), (-1, -3)
        ]
        
        for y in range(3, h - 3):
            for x in range(3, w - 3):
                center = image[y, x]
                
                # Quick check: pixels 1, 5, 9, 13 (top, right, bottom, left)
                p1 = image[y - 3, x]
                p5 = image[y, x + 3]
                p9 = image[y + 3, x]
                p13 = image[y, x - 3]
                
                # At least 3 of these must be brighter or darker
                brighter = int(p1 > center + threshold) + int(p5 > center + threshold) + \
                          int(p9 > center + threshold) + int(p13 > center + threshold)
                darker = int(p1 < center - threshold) + int(p5 < center - threshold) + \
                        int(p9 < center - threshold) + int(p13 < center - threshold)
                
                if brighter < 3 and darker < 3:
                    continue
                
                # Full check: need 9 contiguous pixels
                pixels = [image[y + dy, x + dx] for dx, dy in circle]
                
                is_corner = False
                for mode in ['brighter', 'darker']:
                    count = 0
                    max_count = 0
                    
                    for i in range(len(pixels) * 2):
                        p = pixels[i % len(pixels)]
                        if mode == 'brighter':
                            if p > center + threshold:
                                count += 1
                            else:
                                max_count = max(max_count, count)
                                count = 0
                        else:
                            if p < center - threshold:
                                count += 1
                            else:
                                max_count = max(max_count, count)
                                count = 0
                    
                    max_count = max(max_count, count)
                    if max_count >= 9:
                        is_corner = True
                        break
                
                if is_corner:
                    # Response = sum of absolute differences
                    response = sum(abs(p - center) for p in pixels)
                    corners.append((x, y, response))
        
        return corners
    
    def _compute_harris_response(
        self,
        image: np.ndarray,
        corners: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """Compute Harris corner response for corners."""
        # Compute gradients
        Ix = np.zeros_like(image)
        Iy = np.zeros_like(image)
        Ix[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
        Iy[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
        
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        
        # Gaussian blur
        from scipy.ndimage import gaussian_filter
        Ixx = gaussian_filter(Ixx, 1.5)
        Ixy = gaussian_filter(Ixy, 1.5)
        Iyy = gaussian_filter(Iyy, 1.5)
        
        result = []
        k = 0.04  # Harris parameter
        
        for x, y, _ in corners:
            x, y = int(x), int(y)
            det = Ixx[y, x] * Iyy[y, x] - Ixy[y, x] ** 2
            trace = Ixx[y, x] + Iyy[y, x]
            response = det - k * trace ** 2
            result.append((x, y, response))
        
        return result
    
    def _compute_orientation(self, image: np.ndarray, kp: KeyPoint) -> float:
        """Compute keypoint orientation using intensity centroid."""
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        
        h, w = image.shape
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return 0.0
        
        patch = image[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32)
        
        # Compute moments
        m01 = 0.0
        m10 = 0.0
        
        for py in range(patch.shape[0]):
            for px in range(patch.shape[1]):
                cy = py - r
                cx = px - r
                m01 += cy * patch[py, px]
                m10 += cx * patch[py, px]
        
        angle = np.degrees(np.arctan2(m01, m10))
        if angle < 0:
            angle += 360
        
        return angle
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[KeyPoint],
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Compute ORB descriptors.
        
        Args:
            image: Input image
            keypoints: Detected keypoints
            descriptors: Optional output array
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if image.ndim == 3:
            from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Gaussian blur for noise reduction
        from scipy.ndimage import gaussian_filter
        gray = gaussian_filter(gray.astype(np.float32), 2)
        
        h, w = gray.shape
        half_patch = self._patchSize // 2
        
        valid_kps = []
        descs = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            if x - half_patch < 0 or x + half_patch >= w or \
               y - half_patch < 0 or y + half_patch >= h:
                continue
            
            # Compute rotated BRIEF
            angle = np.radians(kp.angle) if kp.angle >= 0 else 0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            desc = np.zeros(32, dtype=np.uint8)  # 256 bits = 32 bytes
            
            for i, (dx1, dy1, dx2, dy2) in enumerate(self._pattern):
                # Rotate pattern
                rx1 = int(cos_a * dx1 - sin_a * dy1)
                ry1 = int(sin_a * dx1 + cos_a * dy1)
                rx2 = int(cos_a * dx2 - sin_a * dy2)
                ry2 = int(sin_a * dx2 + cos_a * dy2)
                
                # Sample points
                p1x = x + rx1
                p1y = y + ry1
                p2x = x + rx2
                p2y = y + ry2
                
                if 0 <= p1x < w and 0 <= p1y < h and 0 <= p2x < w and 0 <= p2y < h:
                    if gray[p1y, p1x] < gray[p2y, p2x]:
                        desc[i // 8] |= (1 << (i % 8))
            
            valid_kps.append(kp)
            descs.append(desc)
        
        if len(descs) == 0:
            return valid_kps, np.array([], dtype=np.uint8)
        
        return valid_kps, np.array(descs, dtype=np.uint8)
    
    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors."""
        keypoints = self.detect(image, mask)
        return self.compute(image, keypoints, descriptors)
    
    # Getters and setters
    def getMaxFeatures(self) -> int:
        return self._nfeatures
    
    def setMaxFeatures(self, maxFeatures: int):
        self._nfeatures = maxFeatures
    
    def getScaleFactor(self) -> float:
        return self._scaleFactor
    
    def setScaleFactor(self, scaleFactor: float):
        self._scaleFactor = scaleFactor
    
    def getNLevels(self) -> int:
        return self._nlevels
    
    def setNLevels(self, nlevels: int):
        self._nlevels = nlevels
    
    def getEdgeThreshold(self) -> int:
        return self._edgeThreshold
    
    def setEdgeThreshold(self, edgeThreshold: int):
        self._edgeThreshold = edgeThreshold
    
    def getFastThreshold(self) -> int:
        return self._fastThreshold
    
    def setFastThreshold(self, fastThreshold: int):
        self._fastThreshold = fastThreshold


class SIFT(Feature2D):
    """Scale-Invariant Feature Transform.
    
    Note: Simplified implementation - for production use, consider using Neurova's SIFT.
    """
    
    def __init__(
        self,
        nfeatures: int = 0,
        nOctaveLayers: int = 3,
        contrastThreshold: float = 0.04,
        edgeThreshold: float = 10,
        sigma: float = 1.6
    ):
        """Initialize SIFT detector.
        
        Args:
            nfeatures: Number of best features to retain (0 = all)
            nOctaveLayers: Number of layers in each octave
            contrastThreshold: Contrast threshold for filtering weak features
            edgeThreshold: Edge threshold for filtering edge-like features
            sigma: Sigma of Gaussian applied to input image
        """
        self._nfeatures = nfeatures
        self._nOctaveLayers = nOctaveLayers
        self._contrastThreshold = contrastThreshold
        self._edgeThreshold = edgeThreshold
        self._sigma = sigma
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect SIFT keypoints."""
        if image.ndim == 3:
            from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32) / 255.0
        
        from scipy.ndimage import gaussian_filter
        
        # Build scale space
        keypoints = []
        h, w = gray.shape
        
        # Number of octaves
        n_octaves = int(np.log2(min(h, w))) - 2
        
        current = gray
        
        for octave in range(min(n_octaves, 4)):  # Limit octaves for speed
            # Build Gaussian pyramid for this octave
            k = 2 ** (1.0 / self._nOctaveLayers)
            sigmas = [self._sigma * (k ** i) for i in range(self._nOctaveLayers + 3)]
            
            gaussians = [gaussian_filter(current, s) for s in sigmas]
            
            # Difference of Gaussians
            dogs = [gaussians[i+1] - gaussians[i] for i in range(len(gaussians)-1)]
            
            # Find extrema
            for i in range(1, len(dogs) - 1):
                for y in range(1, dogs[i].shape[0] - 1):
                    for x in range(1, dogs[i].shape[1] - 1):
                        if mask is not None:
                            scale = 2 ** octave
                            if not mask[int(y * scale), int(x * scale)]:
                                continue
                        
                        val = dogs[i][y, x]
                        
                        # Check if extremum
                        is_max = True
                        is_min = True
                        
                        for di in range(-1, 2):
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    if di == 0 and dy == 0 and dx == 0:
                                        continue
                                    neighbor = dogs[i + di][y + dy, x + dx]
                                    if neighbor >= val:
                                        is_max = False
                                    if neighbor <= val:
                                        is_min = False
                        
                        if not (is_max or is_min):
                            continue
                        
                        # Contrast check
                        if abs(val) < self._contrastThreshold:
                            continue
                        
                        # Edge check (Hessian)
                        dxx = dogs[i][y, x+1] + dogs[i][y, x-1] - 2*dogs[i][y, x]
                        dyy = dogs[i][y+1, x] + dogs[i][y-1, x] - 2*dogs[i][y, x]
                        dxy = (dogs[i][y+1, x+1] - dogs[i][y+1, x-1] - 
                               dogs[i][y-1, x+1] + dogs[i][y-1, x-1]) / 4
                        
                        tr = dxx + dyy
                        det = dxx * dyy - dxy * dxy
                        
                        if det <= 0:
                            continue
                        
                        ratio = tr * tr / det
                        if ratio > (self._edgeThreshold + 1) ** 2 / self._edgeThreshold:
                            continue
                        
                        # Create keypoint
                        scale = 2 ** octave
                        size = sigmas[i] * scale * 2
                        
                        kp = KeyPoint(
                            x=x * scale,
                            y=y * scale,
                            size=size,
                            angle=-1,
                            response=abs(val),
                            octave=octave
                        )
                        keypoints.append(kp)
            
            # Downsample for next octave
            if octave < n_octaves - 1:
                current = gaussians[self._nOctaveLayers][::2, ::2]
        
        # Sort and limit
        keypoints.sort(key=lambda kp: kp.response, reverse=True)
        if self._nfeatures > 0:
            keypoints = keypoints[:self._nfeatures]
        
        # Compute orientations
        for kp in keypoints:
            kp.angle = self._compute_orientation(gray, kp)
        
        return keypoints
    
    def _compute_orientation(self, image: np.ndarray, kp: KeyPoint) -> float:
        """Compute dominant orientation."""
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size * 1.5)
        
        h, w = image.shape
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return 0.0
        
        patch = image[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
        
        # Compute gradients
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            return 0.0
        
        gx = np.zeros_like(patch)
        gy = np.zeros_like(patch)
        gx[:, 1:-1] = (patch[:, 2:] - patch[:, :-2]) / 2
        gy[1:-1, :] = (patch[2:, :] - patch[:-2, :]) / 2
        
        mag = np.sqrt(gx*gx + gy*gy)
        ori = np.degrees(np.arctan2(gy, gx))
        
        # Histogram of orientations
        hist = np.zeros(36)
        
        for py in range(patch.shape[0]):
            for px in range(patch.shape[1]):
                bin_idx = int((ori[py, px] + 180) / 10) % 36
                hist[bin_idx] += mag[py, px]
        
        # Find dominant orientation
        max_bin = np.argmax(hist)
        angle = max_bin * 10 - 180
        
        return angle if angle >= 0 else angle + 360
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[KeyPoint],
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Compute SIFT descriptors."""
        if image.ndim == 3:
            from ..imgproc.color import cvtColor, COLOR_BGR2GRAY
            gray = cvtColor(image, COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = gray.astype(np.float32) / 255.0
        h, w = gray.shape
        
        valid_kps = []
        descs = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            
            if size < 4:
                size = 4
            
            r = size * 2
            
            if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
                continue
            
            patch = gray[y-r:y+r, x-r:x+r]
            
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                continue
            
            # Compute gradients
            gx = np.zeros_like(patch)
            gy = np.zeros_like(patch)
            gx[:, 1:-1] = (patch[:, 2:] - patch[:, :-2]) / 2
            gy[1:-1, :] = (patch[2:, :] - patch[:-2, :]) / 2
            
            mag = np.sqrt(gx*gx + gy*gy)
            ori = np.degrees(np.arctan2(gy, gx)) - kp.angle
            
            # 4x4 grid, 8 orientation bins
            desc = np.zeros(128)
            
            ph, pw = patch.shape
            cell_h = ph // 4
            cell_w = pw // 4
            
            for cy in range(4):
                for cx in range(4):
                    y1, y2 = cy * cell_h, (cy + 1) * cell_h
                    x1, x2 = cx * cell_w, (cx + 1) * cell_w
                    
                    cell_mag = mag[y1:y2, x1:x2]
                    cell_ori = ori[y1:y2, x1:x2]
                    
                    hist = np.zeros(8)
                    for py in range(cell_mag.shape[0]):
                        for px in range(cell_mag.shape[1]):
                            bin_idx = int((cell_ori[py, px] + 180) / 45) % 8
                            hist[bin_idx] += cell_mag[py, px]
                    
                    desc_idx = (cy * 4 + cx) * 8
                    desc[desc_idx:desc_idx+8] = hist
            
            # Normalize
            norm = np.linalg.norm(desc) + 1e-7
            desc = desc / norm
            
            # Threshold and renormalize
            desc = np.minimum(desc, 0.2)
            norm = np.linalg.norm(desc) + 1e-7
            desc = desc / norm
            
            # Convert to uint8
            desc = (desc * 512).astype(np.float32)
            
            valid_kps.append(kp)
            descs.append(desc)
        
        if len(descs) == 0:
            return valid_kps, np.array([], dtype=np.float32)
        
        return valid_kps, np.array(descs, dtype=np.float32)
    
    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors."""
        keypoints = self.detect(image, mask)
        return self.compute(image, keypoints, descriptors)


class AKAZE(Feature2D):
    """Accelerated-KAZE feature detector and descriptor.
    
    Note: Simplified implementation.
    """
    
    def __init__(
        self,
        descriptor_type: int = 5,  # AKAZE_DESCRIPTOR_MLDB
        descriptor_size: int = 0,
        descriptor_channels: int = 3,
        threshold: float = 0.001,
        nOctaves: int = 4,
        nOctaveLayers: int = 4,
        diffusivity: int = 1  # DIFF_PM_G2
    ):
        """Initialize AKAZE detector."""
        self._descriptor_type = descriptor_type
        self._descriptor_size = descriptor_size
        self._descriptor_channels = descriptor_channels
        self._threshold = threshold
        self._nOctaves = nOctaves
        self._nOctaveLayers = nOctaveLayers
        self._diffusivity = diffusivity
        
        # Use ORB as fallback for simplified implementation
        self._orb = ORB(nfeatures=500)
    
    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[KeyPoint]:
        """Detect AKAZE keypoints (simplified - uses ORB)."""
        return self._orb.detect(image, mask)
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[KeyPoint],
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Compute AKAZE descriptors (simplified - uses ORB)."""
        return self._orb.compute(image, keypoints, descriptors)
    
    def detectAndCompute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None
    ) -> Tuple[List[KeyPoint], np.ndarray]:
        """Detect and compute."""
        return self._orb.detectAndCompute(image, mask, descriptors)


# Factory functions
def ORB_create(
    nfeatures: int = 500,
    scaleFactor: float = 1.2,
    nlevels: int = 8,
    edgeThreshold: int = 31,
    firstLevel: int = 0,
    WTA_K: int = 2,
    scoreType: int = 0,
    patchSize: int = 31,
    fastThreshold: int = 20
) -> ORB:
    """Create ORB detector."""
    return ORB(nfeatures, scaleFactor, nlevels, edgeThreshold,
               firstLevel, WTA_K, scoreType, patchSize, fastThreshold)


def SIFT_create(
    nfeatures: int = 0,
    nOctaveLayers: int = 3,
    contrastThreshold: float = 0.04,
    edgeThreshold: float = 10,
    sigma: float = 1.6
) -> SIFT:
    """Create SIFT detector."""
    return SIFT(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)


def AKAZE_create(
    descriptor_type: int = 5,
    descriptor_size: int = 0,
    descriptor_channels: int = 3,
    threshold: float = 0.001,
    nOctaves: int = 4,
    nOctaveLayers: int = 4,
    diffusivity: int = 1
) -> AKAZE:
    """Create AKAZE detector."""
    return AKAZE(descriptor_type, descriptor_size, descriptor_channels,
                 threshold, nOctaves, nOctaveLayers, diffusivity)


# Score types
HARRIS_SCORE = 0
FAST_SCORE = 1

# AKAZE descriptor types
AKAZE_DESCRIPTOR_KAZE_UPRIGHT = 2
AKAZE_DESCRIPTOR_KAZE = 3
AKAZE_DESCRIPTOR_MLDB_UPRIGHT = 4
AKAZE_DESCRIPTOR_MLDB = 5


__all__ = [
    "KeyPoint",
    "Feature2D",
    "ORB",
    "SIFT",
    "AKAZE",
    "ORB_create",
    "SIFT_create",
    "AKAZE_create",
    "HARRIS_SCORE",
    "FAST_SCORE",
    "AKAZE_DESCRIPTOR_KAZE_UPRIGHT",
    "AKAZE_DESCRIPTOR_KAZE",
    "AKAZE_DESCRIPTOR_MLDB_UPRIGHT",
    "AKAZE_DESCRIPTOR_MLDB",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.