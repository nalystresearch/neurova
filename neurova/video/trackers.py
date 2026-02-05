# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Object tracking classes for Neurova.

Provides TrackerMIL, TrackerKCF, TrackerCSRT and legacy tracker API.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np


class Tracker(ABC):
    """Abstract base class for object trackers."""
    
    @abstractmethod
    def init(self, image: np.ndarray, boundingBox: Tuple[int, int, int, int]) -> bool:
        """Initialize tracker with bounding box.
        
        Args:
            image: Initial frame
            boundingBox: Bounding box (x, y, w, h)
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Update tracker with new frame.
        
        Args:
            image: New frame
        
        Returns:
            Tuple of (success, bounding_box)
        """
        pass


class TrackerMIL(Tracker):
    """Multiple Instance Learning tracker.
    
    This tracker uses online MIL (Multiple Instance Learning) to track objects.
    """
    
    class Params:
        """Parameters for TrackerMIL."""
        
        def __init__(self):
            self.samplerInitInRadius = 3.0
            self.samplerInitMaxNegNum = 65
            self.samplerSearchWinSize = 25.0
            self.samplerTrackInRadius = 4.0
            self.samplerTrackMaxPosNum = 100000
            self.samplerTrackMaxNegNum = 65
            self.featureSetNumFeatures = 250
    
    def __init__(self):
        self.params = TrackerMIL.Params()
        self._initialized = False
        self._bbox = (0, 0, 0, 0)
        self._template = None
        self._search_window = 25
    
    @staticmethod
    def create() -> "TrackerMIL":
        """Create TrackerMIL instance."""
        return TrackerMIL()
    
    def init(self, image: np.ndarray, boundingBox: Tuple[int, int, int, int]) -> bool:
        """Initialize tracker."""
        x, y, w, h = [int(v) for v in boundingBox]
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        # Extract template
        h_img, w_img = gray.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return False
        
        self._template = gray[y:y+h, x:x+w].copy()
        self._bbox = (x, y, w, h)
        self._initialized = True
        
        return True
    
    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Update tracker."""
        if not self._initialized:
            return False, self._bbox
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h_img, w_img = gray.shape
        x, y, w, h = self._bbox
        
        # Define search region
        search_x = max(0, x - self._search_window)
        search_y = max(0, y - self._search_window)
        search_x2 = min(w_img, x + w + self._search_window)
        search_y2 = min(h_img, y + h + self._search_window)
        
        search_region = gray[search_y:search_y2, search_x:search_x2]
        
        # Template matching
        best_score = -1
        best_pos = (x, y)
        
        th, tw = self._template.shape
        
        for dy in range(0, search_region.shape[0] - th + 1, 2):
            for dx in range(0, search_region.shape[1] - tw + 1, 2):
                patch = search_region[dy:dy+th, dx:dx+tw]
                
                # Normalized cross-correlation
                t_norm = self._template - np.mean(self._template)
                p_norm = patch - np.mean(patch)
                
                t_std = np.std(t_norm)
                p_std = np.std(p_norm)
                
                if t_std > 0 and p_std > 0:
                    score = np.sum(t_norm * p_norm) / (t_std * p_std * t_norm.size)
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (search_x + dx, search_y + dy)
        
        if best_score > 0.5:
            self._bbox = (best_pos[0], best_pos[1], w, h)
            # Update template with small learning rate
            new_template = gray[best_pos[1]:best_pos[1]+h, best_pos[0]:best_pos[0]+w]
            if new_template.shape == self._template.shape:
                self._template = (0.9 * self._template + 0.1 * new_template).astype(np.uint8)
            return True, self._bbox
        
        return False, self._bbox


class TrackerKCF(Tracker):
    """Kernelized Correlation Filters tracker.
    
    Fast tracker using correlation filters in Fourier domain.
    """
    
    class Params:
        """Parameters for TrackerKCF."""
        
        def __init__(self):
            self.detect_thresh = 0.5
            self.sigma = 0.2
            self.lambda_val = 0.0001
            self.interp_factor = 0.075
            self.output_sigma_factor = 0.1
            self.resize = True
            self.max_patch_size = 80 * 80
            self.split_coeff = True
            self.wrap_kernel = False
            self.desc_npca = 0
            self.desc_pca = 0
            self.compress_feature = True
            self.compressed_size = 2
            self.pca_learning_rate = 0.15
    
    def __init__(self):
        self.params = TrackerKCF.Params()
        self._initialized = False
        self._bbox = (0, 0, 0, 0)
        self._template = None
        self._alpha = None
        self._tmpl = None
    
    @staticmethod
    def create() -> "TrackerKCF":
        """Create TrackerKCF instance."""
        return TrackerKCF()
    
    def init(self, image: np.ndarray, boundingBox: Tuple[int, int, int, int]) -> bool:
        """Initialize tracker."""
        x, y, w, h = [int(v) for v in boundingBox]
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h_img, w_img = gray.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return False
        
        # Extract and store template
        self._template = gray[y:y+h, x:x+w].copy()
        self._bbox = (x, y, w, h)
        
        # Create Gaussian target
        th, tw = h, w
        y_coords, x_coords = np.ogrid[:th, :tw]
        cx, cy = tw // 2, th // 2
        sigma = self.params.output_sigma_factor * min(tw, th)
        self._target = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * sigma**2))
        
        # Initialize filter
        self._train()
        self._initialized = True
        
        return True
    
    def _train(self) -> None:
        """Train the correlation filter."""
        # HOG-like features (simplified)
        f = self._get_features(self._template)
        
        # FFT
        F = np.fft.fft2(f)
        T = np.fft.fft2(self._target)
        
        # Gaussian kernel in Fourier domain
        k = np.fft.fft2(self._gaussian_correlation(f, f))
        
        # Train filter
        self._alpha = T / (k + self.params.lambda_val)
        self._tmpl = f.copy()
    
    def _get_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features from patch."""
        # Cosine window
        h, w = patch.shape
        hann_y = np.hanning(h)
        hann_x = np.hanning(w)
        window = np.outer(hann_y, hann_x)
        
        # Normalize and window
        patch_norm = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)
        return patch_norm * window
    
    def _gaussian_correlation(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute Gaussian correlation."""
        c = np.fft.ifft2(np.fft.fft2(x1) * np.conj(np.fft.fft2(x2)))
        c = np.real(c)
        
        d = np.sum(x1**2) + np.sum(x2**2) - 2 * c
        d = d.clip(0)
        
        return np.exp(-d / (self.params.sigma**2 * x1.size))
    
    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Update tracker."""
        if not self._initialized:
            return False, self._bbox
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h_img, w_img = gray.shape
        x, y, w, h = self._bbox
        
        # Extract search region
        cx, cy = x + w // 2, y + h // 2
        x1 = max(0, cx - w)
        y1 = max(0, cy - h)
        x2 = min(w_img, cx + w)
        y2 = min(h_img, cy + h)
        
        search = gray[y1:y2, x1:x2]
        
        # Resize to template size if needed
        th, tw = self._template.shape
        if search.shape != (th, tw):
            # Simple resize
            y_idx = np.linspace(0, search.shape[0] - 1, th).astype(int)
            x_idx = np.linspace(0, search.shape[1] - 1, tw).astype(int)
            search = search[np.ix_(y_idx, x_idx)]
        
        # Get features
        z = self._get_features(search)
        
        # Correlation
        k = np.fft.fft2(self._gaussian_correlation(z, self._tmpl))
        response = np.real(np.fft.ifft2(self._alpha * k))
        
        # Find peak
        peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
        peak_val = response[peak_y, peak_x]
        
        if peak_val > self.params.detect_thresh:
            # Update position
            dy = peak_y - th // 2
            dx = peak_x - tw // 2
            
            new_cx = cx + dx
            new_cy = cy + dy
            
            new_x = int(new_cx - w // 2)
            new_y = int(new_cy - h // 2)
            
            # Bounds check
            new_x = max(0, min(new_x, w_img - w))
            new_y = max(0, min(new_y, h_img - h))
            
            self._bbox = (new_x, new_y, w, h)
            
            # Update template
            self._template = gray[new_y:new_y+h, new_x:new_x+w].copy()
            if self._template.shape == (th, tw):
                self._train()
            
            return True, self._bbox
        
        return False, self._bbox


class TrackerCSRT(Tracker):
    """Channel and Spatial Reliability Tracker.
    
    More accurate than KCF but slower.
    """
    
    class Params:
        """Parameters for TrackerCSRT."""
        
        def __init__(self):
            self.use_hog = True
            self.use_color_names = True
            self.use_gray = True
            self.use_rgb = False
            self.use_channel_weights = True
            self.use_segmentation = True
            self.window_function = "hann"
            self.kaiser_alpha = 3.75
            self.cheb_attenuation = 45
            self.template_size = 200
            self.gsl_sigma = 1.0
            self.hog_orientations = 9
            self.hog_clip = 0.2
            self.padding = 3.0
            self.filter_lr = 0.02
            self.weights_lr = 0.02
            self.num_hog_channels_used = 18
            self.admm_iterations = 4
            self.histogram_bins = 16
            self.histogram_lr = 0.04
            self.background_ratio = 2
            self.number_of_scales = 33
            self.scale_sigma_factor = 0.25
            self.scale_model_max_area = 512
            self.scale_lr = 0.025
            self.scale_step = 1.02
            self.psr_threshold = 0.035
    
    def __init__(self):
        self.params = TrackerCSRT.Params()
        self._initialized = False
        self._bbox = (0, 0, 0, 0)
        self._template = None
        self._filter = None
    
    @staticmethod
    def create() -> "TrackerCSRT":
        """Create TrackerCSRT instance."""
        return TrackerCSRT()
    
    def init(self, image: np.ndarray, boundingBox: Tuple[int, int, int, int]) -> bool:
        """Initialize tracker."""
        x, y, w, h = [int(v) for v in boundingBox]
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h_img, w_img = gray.shape
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return False
        
        self._template = gray[y:y+h, x:x+w].copy()
        self._bbox = (x, y, w, h)
        
        # Initialize filter (simplified CSRT)
        self._init_filter()
        self._initialized = True
        
        return True
    
    def _init_filter(self) -> None:
        """Initialize correlation filter."""
        th, tw = self._template.shape
        
        # Gaussian target
        sigma = 0.1 * min(tw, th)
        y_coords, x_coords = np.ogrid[:th, :tw]
        cx, cy = tw // 2, th // 2
        self._target = np.exp(-((x_coords - cx)**2 + (y_coords - cy)**2) / (2 * sigma**2))
        
        # Cosine window
        hann_y = np.hanning(th)
        hann_x = np.hanning(tw)
        self._window = np.outer(hann_y, hann_x)
        
        # Initialize filter in Fourier domain
        f = self._template * self._window
        f = (f - np.mean(f)) / (np.std(f) + 1e-5)
        
        F = np.fft.fft2(f)
        T = np.fft.fft2(self._target)
        
        self._filter = T / (np.conj(F) * F + 1e-4)
        self._model = F
    
    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Update tracker."""
        if not self._initialized:
            return False, self._bbox
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        h_img, w_img = gray.shape
        x, y, w, h = self._bbox
        th, tw = self._template.shape
        
        # Search in larger region
        pad = int(max(w, h) * self.params.padding)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        
        search = gray[y1:y2, x1:x2]
        
        # Resize to template size
        if search.shape != (th, tw):
            y_idx = np.linspace(0, search.shape[0] - 1, th).astype(int)
            x_idx = np.linspace(0, search.shape[1] - 1, tw).astype(int)
            search = search[np.ix_(y_idx, x_idx)]
        
        # Apply window and normalize
        z = search * self._window
        z = (z - np.mean(z)) / (np.std(z) + 1e-5)
        
        Z = np.fft.fft2(z)
        
        # Correlation response
        response = np.real(np.fft.ifft2(self._filter * Z))
        
        # Find peak
        peak_y, peak_x = np.unravel_index(np.argmax(response), response.shape)
        peak_val = response[peak_y, peak_x]
        
        # PSR (Peak to Sidelobe Ratio)
        sidelobe = response.copy()
        sidelobe[max(0, peak_y-5):peak_y+5, max(0, peak_x-5):peak_x+5] = 0
        psr = (peak_val - np.mean(sidelobe)) / (np.std(sidelobe) + 1e-5)
        
        if psr > self.params.psr_threshold * 100:
            # Update position
            scale_x = (x2 - x1) / tw
            scale_y = (y2 - y1) / th
            
            dx = (peak_x - tw // 2) * scale_x
            dy = (peak_y - th // 2) * scale_y
            
            new_x = int(x + dx)
            new_y = int(y + dy)
            
            # Bounds check
            new_x = max(0, min(new_x, w_img - w))
            new_y = max(0, min(new_y, h_img - h))
            
            self._bbox = (new_x, new_y, w, h)
            
            # Update filter
            new_template = gray[new_y:new_y+h, new_x:new_x+w]
            if new_template.shape == (th, tw):
                self._template = new_template.copy()
                
                f = self._template * self._window
                f = (f - np.mean(f)) / (np.std(f) + 1e-5)
                F = np.fft.fft2(f)
                T = np.fft.fft2(self._target)
                
                lr = self.params.filter_lr
                self._filter = (1 - lr) * self._filter + lr * T / (np.conj(F) * F + 1e-4)
                self._model = (1 - lr) * self._model + lr * F
            
            return True, self._bbox
        
        return False, self._bbox


# Legacy tracker API (Neurova 3.x style)
def TrackerMIL_create() -> TrackerMIL:
    """Create TrackerMIL instance (legacy API)."""
    return TrackerMIL.create()


def TrackerKCF_create() -> TrackerKCF:
    """Create TrackerKCF instance (legacy API)."""
    return TrackerKCF.create()


def TrackerCSRT_create() -> TrackerCSRT:
    """Create TrackerCSRT instance (legacy API)."""
    return TrackerCSRT.create()


__all__ = [
    # Classes
    "Tracker",
    "TrackerMIL",
    "TrackerKCF",
    "TrackerCSRT",
    # Factory functions
    "TrackerMIL_create",
    "TrackerKCF_create",
    "TrackerCSRT_create",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.