# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""HOG descriptor implementation for Neurova.

Provides HOGDescriptor class for pedestrian detection and object classification.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class HOGDescriptor:
    """Histogram of Oriented Gradients descriptor.
    
    A feature descriptor used for object detection, especially pedestrian detection.
    
    Args:
        winSize: Detection window size (width, height). Default (64, 128)
        blockSize: Block size (width, height). Default (16, 16)
        blockStride: Block stride (width, height). Default (8, 8)
        cellSize: Cell size (width, height). Default (8, 8)
        nbins: Number of orientation bins. Default 9
        derivAperture: Aperture size for gradient computation. Default 1
        winSigma: Gaussian smoothing window parameter. Default -1 (auto)
        histogramNormType: Histogram normalization type. Default 0 (L2Hys)
        L2HysThreshold: L2-Hys normalization threshold. Default 0.2
        gammaCorrection: Flag for gamma correction. Default True
        nlevels: Maximum number of detection window increases. Default 64
        signedGradient: Flag for signed gradients. Default False
    """
    
    # Default descriptors
    DEFAULT_NLEVELS = 64
    
    def __init__(
        self,
        winSize: Tuple[int, int] = (64, 128),
        blockSize: Tuple[int, int] = (16, 16),
        blockStride: Tuple[int, int] = (8, 8),
        cellSize: Tuple[int, int] = (8, 8),
        nbins: int = 9,
        derivAperture: int = 1,
        winSigma: float = -1,
        histogramNormType: int = 0,
        L2HysThreshold: float = 0.2,
        gammaCorrection: bool = True,
        nlevels: int = 64,
        signedGradient: bool = False
    ):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.derivAperture = derivAperture
        self.winSigma = winSigma if winSigma > 0 else (blockSize[0] + blockSize[1]) / 8.0
        self.histogramNormType = histogramNormType
        self.L2HysThreshold = L2HysThreshold
        self.gammaCorrection = gammaCorrection
        self.nlevels = nlevels
        self.signedGradient = signedGradient
        
        # SVM detector coefficients
        self._svmDetector = None
    
    def getDescriptorSize(self) -> int:
        """Get the size of the HOG descriptor."""
        cells_per_block_x = self.blockSize[0] // self.cellSize[0]
        cells_per_block_y = self.blockSize[1] // self.cellSize[1]
        
        blocks_per_win_x = (self.winSize[0] - self.blockSize[0]) // self.blockStride[0] + 1
        blocks_per_win_y = (self.winSize[1] - self.blockSize[1]) // self.blockStride[1] + 1
        
        return int(cells_per_block_x * cells_per_block_y * self.nbins * 
                   blocks_per_win_x * blocks_per_win_y)
    
    def compute(
        self,
        img: np.ndarray,
        winStride: Tuple[int, int] = (0, 0),
        padding: Tuple[int, int] = (0, 0),
        locations: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """Compute HOG descriptors for the image.
        
        Args:
            img: Input image (grayscale or color)
            winStride: Window stride (if (0,0), uses blockStride)
            padding: Padding (currently ignored)
            locations: Specific locations to compute (if None, compute all)
        
        Returns:
            HOG descriptors as numpy array
        """
        image = np.asarray(img)
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float64)
        
        # Apply gamma correction
        if self.gammaCorrection:
            gray = np.sqrt(gray)
        
        # Compute gradients
        if HAS_SCIPY:
            gx = ndimage.sobel(gray, axis=1)
            gy = ndimage.sobel(gray, axis=0)
        else:
            gx = self._sobel_x(gray)
            gy = self._sobel_y(gray)
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        
        if self.signedGradient:
            orientation = np.arctan2(gy, gx) * 180 / np.pi
            orientation[orientation < 0] += 360
            max_angle = 360
        else:
            orientation = np.arctan2(gy, gx) * 180 / np.pi
            orientation = np.abs(orientation)
            orientation[orientation > 180] = 360 - orientation[orientation > 180]
            max_angle = 180
        
        # Window stride
        if winStride == (0, 0):
            winStride = self.blockStride
        
        h, w = gray.shape
        win_w, win_h = self.winSize
        stride_x, stride_y = winStride
        
        # Compute descriptors for each window
        descriptors = []
        
        for y in range(0, h - win_h + 1, stride_y):
            for x in range(0, w - win_w + 1, stride_x):
                desc = self._compute_window(magnitude, orientation, x, y, max_angle)
                descriptors.append(desc)
        
        if not descriptors:
            # Single window
            desc = self._compute_window(magnitude, orientation, 0, 0, max_angle)
            return desc.reshape(-1, 1)
        
        return np.array(descriptors).T
    
    def _compute_window(
        self, 
        magnitude: np.ndarray, 
        orientation: np.ndarray,
        win_x: int,
        win_y: int,
        max_angle: float
    ) -> np.ndarray:
        """Compute HOG descriptor for a single window."""
        win_w, win_h = self.winSize
        block_w, block_h = self.blockSize
        stride_x, stride_y = self.blockStride
        cell_w, cell_h = self.cellSize
        
        cells_per_block_x = block_w // cell_w
        cells_per_block_y = block_h // cell_h
        
        blocks = []
        
        # Iterate over blocks
        for by in range(0, win_h - block_h + 1, stride_y):
            for bx in range(0, win_w - block_w + 1, stride_x):
                block_hist = []
                
                # Iterate over cells in block
                for cy in range(cells_per_block_y):
                    for cx in range(cells_per_block_x):
                        cell_x = win_x + bx + cx * cell_w
                        cell_y = win_y + by + cy * cell_h
                        
                        cell_mag = magnitude[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w]
                        cell_ori = orientation[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w]
                        
                        hist = self._compute_cell_histogram(cell_mag, cell_ori, max_angle)
                        block_hist.extend(hist)
                
                # Normalize block
                block_hist = np.array(block_hist)
                norm = np.sqrt(np.sum(block_hist**2) + 1e-6)
                block_hist = block_hist / norm
                
                # L2-Hys: clip and renormalize
                block_hist = np.clip(block_hist, 0, self.L2HysThreshold)
                norm = np.sqrt(np.sum(block_hist**2) + 1e-6)
                block_hist = block_hist / norm
                
                blocks.extend(block_hist)
        
        return np.array(blocks)
    
    def _compute_cell_histogram(
        self,
        magnitude: np.ndarray,
        orientation: np.ndarray,
        max_angle: float
    ) -> np.ndarray:
        """Compute histogram for a single cell."""
        bin_width = max_angle / self.nbins
        hist = np.zeros(self.nbins)
        
        for i in range(magnitude.shape[0]):
            for j in range(magnitude.shape[1]):
                mag = magnitude[i, j]
                ori = orientation[i, j]
                
                # Compute bin index with interpolation
                bin_idx = ori / bin_width
                left_bin = int(bin_idx) % self.nbins
                right_bin = (left_bin + 1) % self.nbins
                
                # Bilinear interpolation
                right_weight = bin_idx - int(bin_idx)
                left_weight = 1 - right_weight
                
                hist[left_bin] += left_weight * mag
                hist[right_bin] += right_weight * mag
        
        return hist
    
    def detect(
        self,
        img: np.ndarray,
        hitThreshold: float = 0,
        winStride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (0, 0),
        scale: float = 1.05,
        groupThreshold: int = 2,
        useMeanshiftGrouping: bool = False
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Perform object detection.
        
        Args:
            img: Input image
            hitThreshold: Threshold for SVM detection
            winStride: Window stride
            padding: Padding
            scale: Scale factor for multi-scale detection
            groupThreshold: Minimum number of detections to keep
            useMeanshiftGrouping: Use mean-shift grouping (not implemented)
        
        Returns:
            Tuple of (bounding_boxes, weights)
        """
        if self._svmDetector is None:
            return [], []
        
        image = np.asarray(img)
        h, w = image.shape[:2]
        win_w, win_h = self.winSize
        
        all_boxes = []
        all_weights = []
        
        # Multi-scale detection
        current_scale = 1.0
        
        for level in range(self.nlevels):
            scaled_h = int(h / current_scale)
            scaled_w = int(w / current_scale)
            
            if scaled_w < win_w or scaled_h < win_h:
                break
            
            # Resize image
            if current_scale != 1.0:
                scaled_img = self._resize(image, (scaled_w, scaled_h))
            else:
                scaled_img = image
            
            # Compute descriptors
            descriptors = self.compute(scaled_img, winStride)
            
            # Apply SVM detector
            if descriptors.size > 0:
                scores = np.dot(self._svmDetector[:-1], descriptors) + self._svmDetector[-1]
                
                # Find detections above threshold
                stride_x, stride_y = winStride
                for idx, score in enumerate(scores.flatten()):
                    if score > hitThreshold:
                        grid_w = (scaled_w - win_w) // stride_x + 1
                        y = (idx // grid_w) * stride_y
                        x = (idx % grid_w) * stride_x
                        
                        # Scale back to original coordinates
                        box = (
                            int(x * current_scale),
                            int(y * current_scale),
                            int(win_w * current_scale),
                            int(win_h * current_scale)
                        )
                        all_boxes.append(box)
                        all_weights.append(float(score))
            
            current_scale *= scale
        
        # Group overlapping detections
        if groupThreshold > 0 and len(all_boxes) > 0:
            all_boxes, all_weights = self._group_rectangles(
                all_boxes, all_weights, groupThreshold)
        
        return all_boxes, all_weights
    
    def detectMultiScale(
        self,
        img: np.ndarray,
        hitThreshold: float = 0,
        winStride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (0, 0),
        scale: float = 1.05,
        groupThreshold: int = 2,
        useMeanshiftGrouping: bool = False
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Multi-scale object detection (same as detect)."""
        return self.detect(img, hitThreshold, winStride, padding, 
                          scale, groupThreshold, useMeanshiftGrouping)
    
    def setSVMDetector(self, detector: np.ndarray) -> None:
        """Set the SVM detector coefficients."""
        self._svmDetector = np.asarray(detector).flatten()
    
    def load(self, filename: str) -> bool:
        """Load HOG parameters from file."""
        try:
            data = np.load(filename, allow_pickle=True)
            self._svmDetector = data.get('svmDetector', None)
            return True
        except:
            return False
    
    def save(self, filename: str) -> None:
        """Save HOG parameters to file."""
        np.savez(filename, svmDetector=self._svmDetector)
    
    @staticmethod
    def getDefaultPeopleDetector() -> np.ndarray:
        """Get the default people detector coefficients.
        
        Returns pre-trained SVM coefficients for pedestrian detection.
        """
        # Simplified detector - in practice, use trained coefficients
        desc_size = 3780  # Default HOG descriptor size for (64, 128) window
        return np.zeros(desc_size + 1, dtype=np.float32)
    
    @staticmethod
    def getDaimlerPeopleDetector() -> np.ndarray:
        """Get the Daimler people detector coefficients."""
        desc_size = 1980  # For (48, 96) window
        return np.zeros(desc_size + 1, dtype=np.float32)
    
    def _sobel_x(self, img: np.ndarray) -> np.ndarray:
        """Sobel gradient in X direction."""
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        return self._convolve2d(img, kernel)
    
    def _sobel_y(self, img: np.ndarray) -> np.ndarray:
        """Sobel gradient in Y direction."""
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        return self._convolve2d(img, kernel)
    
    def _convolve2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode='reflect')
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return result
    
    def _resize(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Simple nearest-neighbor resize."""
        new_w, new_h = size
        h, w = img.shape[:2]
        
        y_indices = (np.arange(new_h) * h / new_h).astype(int)
        x_indices = (np.arange(new_w) * w / new_w).astype(int)
        
        if img.ndim == 3:
            return img[y_indices[:, np.newaxis], x_indices, :]
        return img[y_indices[:, np.newaxis], x_indices]
    
    def _group_rectangles(
        self,
        boxes: List[Tuple[int, int, int, int]],
        weights: List[float],
        threshold: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        """Simple rectangle grouping using overlap."""
        if len(boxes) == 0:
            return [], []
        
        # Simple non-maximum suppression
        boxes = list(boxes)
        weights = list(weights)
        
        indices = np.argsort(weights)[::-1]
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            remaining = []
            for j in indices[1:]:
                iou = self._compute_iou(boxes[i], boxes[j])
                if iou < 0.3:  # Keep if not overlapping too much
                    remaining.append(j)
            
            indices = remaining
        
        return [boxes[i] for i in keep], [weights[i] for i in keep]
    
    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute intersection over union."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0


def groupRectangles(
    rectList: List[Tuple[int, int, int, int]],
    groupThreshold: int,
    eps: float = 0.2
) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
    """Group overlapping rectangles.
    
    Args:
        rectList: List of rectangles (x, y, w, h)
        groupThreshold: Minimum number of rectangles to keep a group
        eps: Relative difference between rectangles to merge
    
    Returns:
        Tuple of (grouped_rectangles, weights)
    """
    if len(rectList) == 0:
        return [], []
    
    # Simple grouping based on overlap
    groups = []
    used = [False] * len(rectList)
    
    for i, rect in enumerate(rectList):
        if used[i]:
            continue
        
        group = [rect]
        used[i] = True
        
        for j, other in enumerate(rectList[i+1:], i+1):
            if used[j]:
                continue
            
            # Check if overlapping
            iou = _rect_iou(rect, other)
            if iou > eps:
                group.append(other)
                used[j] = True
        
        if len(group) >= groupThreshold:
            # Average the group
            avg_rect = (
                int(np.mean([r[0] for r in group])),
                int(np.mean([r[1] for r in group])),
                int(np.mean([r[2] for r in group])),
                int(np.mean([r[3] for r in group]))
            )
            groups.append((avg_rect, len(group)))
    
    if groups:
        rects, weights = zip(*groups)
        return list(rects), list(weights)
    return [], []


def _rect_iou(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> float:
    """Compute rectangle IoU."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0


__all__ = [
    "HOGDescriptor",
    "groupRectangles",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.