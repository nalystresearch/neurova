# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.video.background - Background Subtraction

Provides Neurova background subtractor implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class BackgroundSubtractor:
    """Base class for background subtractors."""
    
    def apply(
        self,
        image: np.ndarray,
        fgmask: Optional[np.ndarray] = None,
        learningRate: float = -1
    ) -> np.ndarray:
        """Apply background subtraction."""
        raise NotImplementedError
    
    def getBackgroundImage(self, backgroundImage: Optional[np.ndarray] = None) -> np.ndarray:
        """Get current background model."""
        raise NotImplementedError


class BackgroundSubtractorMOG2(BackgroundSubtractor):
    """Gaussian Mixture-based Background/Foreground Segmentation.
    
    Based on Zivkovic's papers.
    """
    
    def __init__(
        self,
        history: int = 500,
        varThreshold: float = 16,
        detectShadows: bool = True
    ):
        """Initialize MOG2 background subtractor.
        
        Args:
            history: Length of history
            varThreshold: Threshold on squared Mahalanobis distance
            detectShadows: Whether to detect shadows
        """
        self._history = history
        self._varThreshold = varThreshold
        self._detectShadows = detectShadows
        
        self._nMixtures = 5
        self._backgroundRatio = 0.9
        self._varMin = 4
        self._varMax = 5 * self._varThreshold
        self._varInit = 15
        self._complexityReductionThreshold = 0.05
        self._shadowValue = 127
        self._shadowThreshold = 0.5
        
        self._initialized = False
        self._frame_count = 0
        self._background = None
        
        # Per-pixel GMM parameters
        self._weights = None
        self._means = None
        self._variances = None
    
    def apply(
        self,
        image: np.ndarray,
        fgmask: Optional[np.ndarray] = None,
        learningRate: float = -1
    ) -> np.ndarray:
        """Apply background subtraction.
        
        Args:
            image: Input image (color or grayscale)
            fgmask: Optional output mask
            learningRate: Learning rate (-1 for auto)
        
        Returns:
            Foreground mask (0=bg, 255=fg, 127=shadow if enabled)
        """
        if image.ndim == 3:
            img = image.astype(np.float32)
            is_color = True
        else:
            img = image.astype(np.float32)[:, :, np.newaxis]
            is_color = False
        
        h, w, c = img.shape
        
        # Auto learning rate
        if learningRate < 0:
            alpha = 1.0 / min(self._frame_count + 1, self._history)
        else:
            alpha = learningRate
        
        if not self._initialized:
            # Initialize with first frame
            self._weights = np.zeros((h, w, self._nMixtures), dtype=np.float32)
            self._weights[:, :, 0] = 1.0
            
            self._means = np.zeros((h, w, self._nMixtures, c), dtype=np.float32)
            self._means[:, :, 0, :] = img
            
            self._variances = np.full((h, w, self._nMixtures), 
                                     self._varInit, dtype=np.float32)
            
            self._background = img.copy()
            self._initialized = True
            self._frame_count = 1
            
            return np.zeros((h, w), dtype=np.uint8)
        
        self._frame_count += 1
        
        # Foreground mask
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process each pixel (vectorized for efficiency)
        for k in range(self._nMixtures):
            # Distance to this component
            diff = img - self._means[:, :, k, :]
            dist_sq = np.sum(diff ** 2, axis=2) / (self._variances[:, :, k] + 1e-6)
            
            # Check if matches
            match = dist_sq < self._varThreshold
            
            # Update matching components
            weight_update = alpha * (1 - self._weights[:, :, k])
            self._weights[:, :, k] = np.where(
                match, 
                self._weights[:, :, k] + weight_update,
                (1 - alpha) * self._weights[:, :, k]
            )
            
            rho = alpha / (self._weights[:, :, k] + 1e-6)
            rho = np.clip(rho, 0, 1)
            
            for ch in range(c):
                self._means[:, :, k, ch] = np.where(
                    match,
                    (1 - rho) * self._means[:, :, k, ch] + rho * img[:, :, ch],
                    self._means[:, :, k, ch]
                )
            
            new_var = np.sum(diff ** 2, axis=2)
            self._variances[:, :, k] = np.where(
                match,
                (1 - rho) * self._variances[:, :, k] + rho * new_var,
                self._variances[:, :, k]
            )
            self._variances[:, :, k] = np.clip(
                self._variances[:, :, k], 
                self._varMin, 
                self._varMax
            )
        
        # Normalize weights
        weight_sum = np.sum(self._weights, axis=2, keepdims=True)
        self._weights = self._weights / (weight_sum + 1e-6)
        
        # Determine background/foreground
        # Sort components by weight/variance (background likely)
        sorted_indices = np.argsort(
            -self._weights / (np.sqrt(self._variances) + 1e-6), 
            axis=2
        )
        
        # Cumulative weight check for background model
        cumsum = np.cumsum(
            np.take_along_axis(self._weights, sorted_indices, axis=2), 
            axis=2
        )
        is_background = cumsum < self._backgroundRatio
        
        # Check pixel matching any background component
        for k in range(self._nMixtures):
            diff = img - self._means[:, :, k, :]
            dist_sq = np.sum(diff ** 2, axis=2) / (self._variances[:, :, k] + 1e-6)
            match = dist_sq < self._varThreshold
            
            # This is foreground if it doesn't match any background component
            # For simplicity, mark as background if matches top component
            if k == 0:
                fg_mask = np.where(match, 0, 255).astype(np.uint8)
        
        # Shadow detection
        if self._detectShadows:
            # Simple shadow detection based on brightness ratio
            bg_intensity = np.sum(self._means[:, :, 0, :], axis=2) / c
            img_intensity = np.sum(img, axis=2) / c
            
            ratio = img_intensity / (bg_intensity + 1e-6)
            is_shadow = (ratio > self._shadowThreshold) & (ratio < 1.0) & (fg_mask > 0)
            
            fg_mask = np.where(is_shadow, self._shadowValue, fg_mask)
        
        # Update background image
        self._background = self._means[:, :, 0, :]
        
        if fgmask is not None:
            np.copyto(fgmask, fg_mask)
            return fgmask
        
        return fg_mask
    
    def getBackgroundImage(self, backgroundImage: Optional[np.ndarray] = None) -> np.ndarray:
        """Get current background model."""
        if self._background is None:
            raise RuntimeError("Background subtractor not initialized")
        
        bg = self._background.astype(np.uint8)
        if bg.shape[2] == 1:
            bg = bg[:, :, 0]
        
        if backgroundImage is not None:
            np.copyto(backgroundImage, bg)
            return backgroundImage
        
        return bg
    
    def getHistory(self) -> int:
        return self._history
    
    def setHistory(self, history: int):
        self._history = history
    
    def getNMixtures(self) -> int:
        return self._nMixtures
    
    def setNMixtures(self, nmixtures: int):
        self._nMixtures = nmixtures
    
    def getBackgroundRatio(self) -> float:
        return self._backgroundRatio
    
    def setBackgroundRatio(self, ratio: float):
        self._backgroundRatio = ratio
    
    def getVarThreshold(self) -> float:
        return self._varThreshold
    
    def setVarThreshold(self, varThreshold: float):
        self._varThreshold = varThreshold
    
    def getDetectShadows(self) -> bool:
        return self._detectShadows
    
    def setDetectShadows(self, detectShadows: bool):
        self._detectShadows = detectShadows
    
    def getShadowValue(self) -> int:
        return self._shadowValue
    
    def setShadowValue(self, value: int):
        self._shadowValue = value
    
    def getShadowThreshold(self) -> float:
        return self._shadowThreshold
    
    def setShadowThreshold(self, threshold: float):
        self._shadowThreshold = threshold


class BackgroundSubtractorKNN(BackgroundSubtractor):
    """K-nearest neighbors-based Background/Foreground Segmentation."""
    
    def __init__(
        self,
        history: int = 500,
        dist2Threshold: float = 400.0,
        detectShadows: bool = True
    ):
        """Initialize KNN background subtractor.
        
        Args:
            history: Length of history
            dist2Threshold: Threshold on squared distance
            detectShadows: Whether to detect shadows
        """
        self._history = history
        self._dist2Threshold = dist2Threshold
        self._detectShadows = detectShadows
        
        self._kNNSamples = 7
        self._nSamples = 10
        self._shadowValue = 127
        self._shadowThreshold = 0.5
        
        self._initialized = False
        self._frame_count = 0
        self._samples = None
        self._background = None
    
    def apply(
        self,
        image: np.ndarray,
        fgmask: Optional[np.ndarray] = None,
        learningRate: float = -1
    ) -> np.ndarray:
        """Apply background subtraction.
        
        Args:
            image: Input image
            fgmask: Optional output mask
            learningRate: Learning rate
        
        Returns:
            Foreground mask
        """
        if image.ndim == 3:
            img = image.astype(np.float32)
        else:
            img = image.astype(np.float32)[:, :, np.newaxis]
        
        h, w, c = img.shape
        
        if learningRate < 0:
            alpha = 1.0 / min(self._frame_count + 1, self._history)
        else:
            alpha = learningRate
        
        if not self._initialized:
            # Initialize sample buffer
            self._samples = np.zeros((h, w, self._nSamples, c), dtype=np.float32)
            self._samples[:, :, 0, :] = img
            
            self._background = img.copy()
            self._initialized = True
            self._frame_count = 1
            
            return np.zeros((h, w), dtype=np.uint8)
        
        self._frame_count += 1
        
        # Compute distances to all samples
        dist_sq = np.sum((img[:, :, np.newaxis, :] - self._samples) ** 2, axis=3)
        
        # Count neighbors within threshold
        neighbors = np.sum(dist_sq < self._dist2Threshold, axis=2)
        
        # Foreground if not enough neighbors
        fg_mask = np.where(neighbors < self._kNNSamples, 255, 0).astype(np.uint8)
        
        # Update samples with probability
        update_mask = np.random.random((h, w)) < alpha
        sample_idx = np.random.randint(0, self._nSamples, (h, w))
        
        for i in range(self._nSamples):
            mask = update_mask & (sample_idx == i) & (fg_mask == 0)
            for ch in range(c):
                self._samples[:, :, i, ch] = np.where(
                    mask, img[:, :, ch], self._samples[:, :, i, ch]
                )
        
        # Background is median of samples
        self._background = np.median(self._samples, axis=2)
        
        # Shadow detection
        if self._detectShadows:
            bg_intensity = np.mean(self._background, axis=2)
            img_intensity = np.mean(img, axis=2)
            
            ratio = img_intensity / (bg_intensity + 1e-6)
            is_shadow = (ratio > self._shadowThreshold) & (ratio < 1.0) & (fg_mask > 0)
            
            fg_mask = np.where(is_shadow, self._shadowValue, fg_mask)
        
        if fgmask is not None:
            np.copyto(fgmask, fg_mask)
            return fgmask
        
        return fg_mask
    
    def getBackgroundImage(self, backgroundImage: Optional[np.ndarray] = None) -> np.ndarray:
        """Get current background model."""
        if self._background is None:
            raise RuntimeError("Background subtractor not initialized")
        
        bg = self._background.astype(np.uint8)
        if bg.shape[2] == 1:
            bg = bg[:, :, 0]
        
        if backgroundImage is not None:
            np.copyto(backgroundImage, bg)
            return backgroundImage
        
        return bg
    
    def getHistory(self) -> int:
        return self._history
    
    def setHistory(self, history: int):
        self._history = history
    
    def getDist2Threshold(self) -> float:
        return self._dist2Threshold
    
    def setDist2Threshold(self, threshold: float):
        self._dist2Threshold = threshold
    
    def getDetectShadows(self) -> bool:
        return self._detectShadows
    
    def setDetectShadows(self, detectShadows: bool):
        self._detectShadows = detectShadows
    
    def getShadowValue(self) -> int:
        return self._shadowValue
    
    def setShadowValue(self, value: int):
        self._shadowValue = value
    
    def getkNNSamples(self) -> int:
        return self._kNNSamples
    
    def setkNNSamples(self, nkNN: int):
        self._kNNSamples = nkNN


def createBackgroundSubtractorMOG2(
    history: int = 500,
    varThreshold: float = 16,
    detectShadows: bool = True
) -> BackgroundSubtractorMOG2:
    """Create MOG2 background subtractor.
    
    Args:
        history: Length of history
        varThreshold: Threshold on squared Mahalanobis distance
        detectShadows: Whether to detect shadows
    
    Returns:
        BackgroundSubtractorMOG2 instance
    """
    return BackgroundSubtractorMOG2(history, varThreshold, detectShadows)


def createBackgroundSubtractorKNN(
    history: int = 500,
    dist2Threshold: float = 400.0,
    detectShadows: bool = True
) -> BackgroundSubtractorKNN:
    """Create KNN background subtractor.
    
    Args:
        history: Length of history
        dist2Threshold: Threshold on squared distance
        detectShadows: Whether to detect shadows
    
    Returns:
        BackgroundSubtractorKNN instance
    """
    return BackgroundSubtractorKNN(history, dist2Threshold, detectShadows)


__all__ = [
    "BackgroundSubtractor",
    "BackgroundSubtractorMOG2",
    "BackgroundSubtractorKNN",
    "createBackgroundSubtractorMOG2",
    "createBackgroundSubtractorKNN",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.