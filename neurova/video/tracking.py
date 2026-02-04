# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Tracking algorithms for Neurova.

Provides CamShift, meanShift, and KalmanFilter for object tracking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def CamShift(
    probImage: np.ndarray,
    window: Tuple[int, int, int, int],
    criteria: Tuple[int, int, float]
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float], float], Tuple[int, int, int, int]]:
    """Find object center, size, and orientation using CAMshift algorithm.
    
    CAMshift (Continuously Adaptive Mean Shift) automatically adjusts
    the search window size based on the object size in the probability image.
    
    Args:
        probImage: Back projection of the object histogram
        window: Initial search window (x, y, width, height)
        criteria: Stop criteria (type, maxCount, epsilon)
    
    Returns:
        Tuple of (RotatedRect, updated_window)
        RotatedRect is ((center_x, center_y), (width, height), angle)
    """
    x, y, w, h = window
    _, max_iter, epsilon = criteria
    
    # Run meanShift first to find the location
    _, new_window = meanShift(probImage, window, criteria)
    
    x, y, w, h = new_window
    
    # Extract the window from probability image
    img_h, img_w = probImage.shape[:2]
    
    # Clamp to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    
    roi = probImage[y:y+h, x:x+w].astype(np.float64)
    
    if roi.size == 0:
        center = (x + w/2, y + h/2)
        return ((center, (float(w), float(h)), 0.0), (x, y, w, h))
    
    # Compute moments for orientation
    M = _compute_moments(roi)
    
    if M['m00'] > 0:
        # Center of mass within ROI
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        # Compute orientation from second moments
        mu20 = M['m20'] / M['m00'] - cx**2
        mu02 = M['m02'] / M['m00'] - cy**2
        mu11 = M['m11'] / M['m00'] - cx * cy
        
        # Orientation angle
        if abs(mu20 - mu02) > 1e-10:
            angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) * 180 / np.pi
        else:
            angle = 0.0
        
        # Adjust window size based on moments
        s = 2.0 * np.sqrt(M['m00'])
        new_w = max(1, int(s * 1.2))
        new_h = max(1, int(s * 1.2))
        
        # Center in image coordinates
        center_x = x + cx
        center_y = y + cy
        
        # Update window
        new_x = max(0, int(center_x - new_w/2))
        new_y = max(0, int(center_y - new_h/2))
        new_w = min(new_w, img_w - new_x)
        new_h = min(new_h, img_h - new_y)
        
        center = (float(center_x), float(center_y))
        size = (float(new_w), float(new_h))
        
        return ((center, size, float(angle)), (new_x, new_y, new_w, new_h))
    
    # Fallback
    center = (x + w/2, y + h/2)
    return ((center, (float(w), float(h)), 0.0), (x, y, w, h))


def meanShift(
    probImage: np.ndarray,
    window: Tuple[int, int, int, int],
    criteria: Tuple[int, int, float]
) -> Tuple[int, Tuple[int, int, int, int]]:
    """Find object on a back projection image using meanshift.
    
    Args:
        probImage: Back projection of the object histogram
        window: Initial search window (x, y, width, height)
        criteria: Stop criteria (type, maxCount, epsilon)
    
    Returns:
        Tuple of (iterations, updated_window)
    """
    x, y, w, h = window
    _, max_iter, epsilon = criteria
    
    img = np.asarray(probImage, dtype=np.float64)
    img_h, img_w = img.shape[:2]
    
    for iteration in range(max_iter):
        # Clamp window to image bounds
        x = max(0, min(x, img_w - w))
        y = max(0, min(y, img_h - h))
        
        # Extract window
        roi = img[y:y+h, x:x+w]
        
        if roi.size == 0:
            break
        
        # Compute center of mass
        total = np.sum(roi)
        
        if total < 1e-10:
            break
        
        # Create coordinate grids
        yy, xx = np.mgrid[0:h, 0:w]
        
        cx = np.sum(xx * roi) / total
        cy = np.sum(yy * roi) / total
        
        # New center in image coordinates
        new_x = int(x + cx - w/2)
        new_y = int(y + cy - h/2)
        
        # Check convergence
        dx = new_x - x
        dy = new_y - y
        
        if dx*dx + dy*dy < epsilon*epsilon:
            return iteration + 1, (new_x, new_y, w, h)
        
        x, y = new_x, new_y
    
    return max_iter, (x, y, w, h)


class KalmanFilter:
    """Kalman filter for state estimation and prediction.
    
    The Kalman filter is an efficient recursive filter that estimates
    the internal state of a linear dynamic system from a series of
    noisy measurements.
    
    Args:
        dynamParams: Dimensionality of the state
        measureParams: Dimensionality of the measurement
        controlParams: Dimensionality of the control vector (default 0)
        type: Data type (ignored, uses float64)
    """
    
    def __init__(
        self,
        dynamParams: int,
        measureParams: int,
        controlParams: int = 0,
        type: int = 5  # CV_32F
    ):
        self.dynamParams = dynamParams
        self.measureParams = measureParams
        self.controlParams = controlParams
        
        # State vector
        self.statePre = np.zeros((dynamParams, 1), dtype=np.float64)
        self.statePost = np.zeros((dynamParams, 1), dtype=np.float64)
        
        # Transition matrix (A)
        self.transitionMatrix = np.eye(dynamParams, dtype=np.float64)
        
        # Control matrix (B)
        self.controlMatrix = np.zeros((dynamParams, max(controlParams, 1)), dtype=np.float64)
        
        # Measurement matrix (H)
        self.measurementMatrix = np.zeros((measureParams, dynamParams), dtype=np.float64)
        
        # Process noise covariance (Q)
        self.processNoiseCov = np.eye(dynamParams, dtype=np.float64)
        
        # Measurement noise covariance (R)
        self.measurementNoiseCov = np.eye(measureParams, dtype=np.float64)
        
        # Error covariance matrices
        self.errorCovPre = np.zeros((dynamParams, dynamParams), dtype=np.float64)
        self.errorCovPost = np.zeros((dynamParams, dynamParams), dtype=np.float64)
        
        # Kalman gain
        self.gain = np.zeros((dynamParams, measureParams), dtype=np.float64)
        
        # Temporary matrices
        self._temp1 = np.zeros((dynamParams, dynamParams), dtype=np.float64)
        self._temp2 = np.zeros((measureParams, dynamParams), dtype=np.float64)
        self._temp3 = np.zeros((measureParams, measureParams), dtype=np.float64)
        self._temp4 = np.zeros((measureParams, dynamParams), dtype=np.float64)
        self._temp5 = np.zeros((measureParams, 1), dtype=np.float64)
    
    def predict(self, control: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute a predicted state.
        
        Args:
            control: Optional control vector
        
        Returns:
            Predicted state vector
        """
        # statePre = A * statePost
        self.statePre = self.transitionMatrix @ self.statePost
        
        # Add control input: statePre += B * control
        if control is not None and self.controlParams > 0:
            control = np.asarray(control).reshape(-1, 1)
            self.statePre += self.controlMatrix @ control
        
        # errorCovPre = A * errorCovPost * A' + Q
        self.errorCovPre = (self.transitionMatrix @ self.errorCovPost @ 
                           self.transitionMatrix.T + self.processNoiseCov)
        
        # Copy to statePost for correct usage
        self.statePost = self.statePre.copy()
        
        return self.statePre.copy()
    
    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Update the predicted state from the measurement.
        
        Args:
            measurement: Measured state vector
        
        Returns:
            Corrected state vector
        """
        measurement = np.asarray(measurement).reshape(-1, 1)
        
        # temp2 = H * errorCovPre
        self._temp2 = self.measurementMatrix @ self.errorCovPre
        
        # temp3 = H * errorCovPre * H' + R
        self._temp3 = self._temp2 @ self.measurementMatrix.T + self.measurementNoiseCov
        
        # Kalman gain: K = errorCovPre * H' * inv(temp3)
        try:
            temp3_inv = np.linalg.inv(self._temp3)
            self.gain = self.errorCovPre @ self.measurementMatrix.T @ temp3_inv
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            temp3_pinv = np.linalg.pinv(self._temp3)
            self.gain = self.errorCovPre @ self.measurementMatrix.T @ temp3_pinv
        
        # statePost = statePre + K * (measurement - H * statePre)
        self._temp5 = measurement - self.measurementMatrix @ self.statePre
        self.statePost = self.statePre + self.gain @ self._temp5
        
        # errorCovPost = (I - K * H) * errorCovPre
        I_KH = np.eye(self.dynamParams) - self.gain @ self.measurementMatrix
        self.errorCovPost = I_KH @ self.errorCovPre
        
        return self.statePost.copy()
    
    def init(
        self,
        statePre: np.ndarray,
        errorCovPost: np.ndarray
    ) -> None:
        """Re-initialize the filter.
        
        Args:
            statePre: Initial state
            errorCovPost: Initial error covariance
        """
        self.statePre = np.asarray(statePre).reshape(-1, 1).astype(np.float64)
        self.statePost = self.statePre.copy()
        self.errorCovPost = np.asarray(errorCovPost).astype(np.float64)
        self.errorCovPre = self.errorCovPost.copy()


def _compute_moments(roi: np.ndarray) -> dict:
    """Compute image moments."""
    h, w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    
    m00 = np.sum(roi)
    m10 = np.sum(xx * roi)
    m01 = np.sum(yy * roi)
    m20 = np.sum(xx * xx * roi)
    m02 = np.sum(yy * yy * roi)
    m11 = np.sum(xx * yy * roi)
    
    return {
        'm00': m00, 'm10': m10, 'm01': m01,
        'm20': m20, 'm02': m02, 'm11': m11
    }


# Termination criteria types
TERM_CRITERIA_EPS = 1
TERM_CRITERIA_MAX_ITER = 2
TERM_CRITERIA_COUNT = 2


__all__ = [
    "CamShift",
    "meanShift",
    "KalmanFilter",
    "TERM_CRITERIA_EPS",
    "TERM_CRITERIA_MAX_ITER",
    "TERM_CRITERIA_COUNT",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.