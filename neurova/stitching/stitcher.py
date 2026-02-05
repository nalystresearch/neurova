# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Image stitching module for Neurova.

Provides Stitcher class and related functions for panorama creation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np


class Stitcher:
    """High level image stitcher class.
    
    Implements image stitching for panorama creation.
    """
    
    class Mode(IntEnum):
        PANORAMA = 0
        SCANS = 1
    
    class Status(IntEnum):
        OK = 0
        ERR_NEED_MORE_IMGS = 1
        ERR_HOMOGRAPHY_EST_FAIL = 2
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    
    def __init__(self, mode: Mode = Mode.PANORAMA):
        """Initialize stitcher.
        
        Args:
            mode: Stitching mode (PANORAMA or SCANS)
        """
        self.mode = mode
        self._registration_resol = 0.6
        self._seam_estimation_resol = 0.1
        self._compositing_resol = -1  # Same as input
        self._pano_confidence_thresh = 1.0
        self._wave_correct_kind = 0  # Horizontal
    
    @staticmethod
    def create(mode: "Stitcher.Mode" = Mode.PANORAMA) -> "Stitcher":
        """Create a Stitcher with given mode.
        
        Args:
            mode: Stitching mode
        
        Returns:
            Stitcher instance
        """
        return Stitcher(mode)
    
    def stitch(
        self,
        images: List[np.ndarray],
        pano: Optional[np.ndarray] = None
    ) -> Tuple["Stitcher.Status", np.ndarray]:
        """Stitch images into a panorama.
        
        Args:
            images: List of input images
            pano: Output panorama (optional)
        
        Returns:
            Tuple of (status, panorama_image)
        """
        if len(images) < 2:
            return Stitcher.Status.ERR_NEED_MORE_IMGS, np.zeros((1, 1, 3), dtype=np.uint8)
        
        try:
            # 1. Feature detection and matching
            features = []
            for img in images:
                kps, descs = self._detect_features(img)
                features.append((kps, descs))
            
            # 2. Pairwise matching
            matches = self._match_features(features)
            
            if len(matches) == 0:
                return Stitcher.Status.ERR_HOMOGRAPHY_EST_FAIL, np.zeros((1, 1, 3), dtype=np.uint8)
            
            # 3. Estimate homographies
            homographies = self._estimate_homographies(images, features, matches)
            
            if len(homographies) == 0:
                return Stitcher.Status.ERR_HOMOGRAPHY_EST_FAIL, np.zeros((1, 1, 3), dtype=np.uint8)
            
            # 4. Warp and blend
            result = self._warp_and_blend(images, homographies)
            
            return Stitcher.Status.OK, result
        
        except Exception as e:
            return Stitcher.Status.ERR_CAMERA_PARAMS_ADJUST_FAIL, np.zeros((1, 1, 3), dtype=np.uint8)
    
    def _detect_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect features in image."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        
        # Scale down for feature detection
        scale = self._registration_resol
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Simple resize
        indices_y = np.linspace(0, h - 1, new_h).astype(int)
        indices_x = np.linspace(0, w - 1, new_w).astype(int)
        scaled = gray[np.ix_(indices_y, indices_x)]
        
        # Harris corners as features
        keypoints = []
        Ix = np.zeros_like(scaled, dtype=np.float32)
        Iy = np.zeros_like(scaled, dtype=np.float32)
        
        Ix[:, 1:-1] = scaled[:, 2:].astype(float) - scaled[:, :-2].astype(float)
        Iy[1:-1, :] = scaled[2:, :].astype(float) - scaled[:-2, :].astype(float)
        
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Box filter
        from scipy import ndimage
        k = np.ones((3, 3)) / 9
        Sxx = ndimage.convolve(Ixx, k)
        Syy = ndimage.convolve(Iyy, k)
        Sxy = ndimage.convolve(Ixy, k)
        
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        response = det - 0.04 * trace * trace
        
        # Find local maxima
        threshold = 0.01 * response.max()
        
        for y in range(5, new_h - 5):
            for x in range(5, new_w - 5):
                if response[y, x] > threshold:
                    is_max = True
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            if dy == 0 and dx == 0:
                                continue
                            if response[y + dy, x + dx] >= response[y, x]:
                                is_max = False
                                break
                        if not is_max:
                            break
                    
                    if is_max:
                        # Scale back to original coordinates
                        keypoints.append((x / scale, y / scale, response[y, x]))
        
        # Sort by response and take top 500
        keypoints.sort(key=lambda k: k[2], reverse=True)
        keypoints = keypoints[:500]
        
        # Compute simple descriptors
        descriptors = []
        for kx, ky, _ in keypoints:
            x, y = int(kx * scale), int(ky * scale)
            if 8 <= x < new_w - 8 and 8 <= y < new_h - 8:
                patch = scaled[y-8:y+8, x-8:x+8].flatten()
                patch = (patch - patch.mean()) / (patch.std() + 1e-10)
                descriptors.append(patch)
            else:
                descriptors.append(np.zeros(256))
        
        return keypoints, np.array(descriptors, dtype=np.float32)
    
    def _match_features(
        self,
        features: List[Tuple[List, np.ndarray]]
    ) -> List[Tuple[int, int, List]]:
        """Match features between image pairs."""
        matches = []
        n = len(features)
        
        for i in range(n):
            for j in range(i + 1, n):
                kps1, desc1 = features[i]
                kps2, desc2 = features[j]
                
                if len(desc1) == 0 or len(desc2) == 0:
                    continue
                
                # Brute force matching with ratio test
                pair_matches = []
                
                for k, d1 in enumerate(desc1):
                    distances = np.linalg.norm(desc2 - d1, axis=1)
                    sorted_idx = np.argsort(distances)
                    
                    if len(sorted_idx) >= 2:
                        if distances[sorted_idx[0]] < 0.75 * distances[sorted_idx[1]]:
                            pair_matches.append((k, sorted_idx[0]))
                
                if len(pair_matches) >= 8:
                    matches.append((i, j, pair_matches))
        
        return matches
    
    def _estimate_homographies(
        self,
        images: List[np.ndarray],
        features: List[Tuple[List, np.ndarray]],
        matches: List[Tuple[int, int, List]]
    ) -> List[np.ndarray]:
        """Estimate homographies between images."""
        n = len(images)
        
        # Use first image as reference
        homographies = [np.eye(3) for _ in range(n)]
        
        for i, j, pair_matches in matches:
            kps1, _ = features[i]
            kps2, _ = features[j]
            
            # Get matched points
            pts1 = np.array([[kps1[m[0]][0], kps1[m[0]][1]] for m in pair_matches])
            pts2 = np.array([[kps2[m[1]][0], kps2[m[1]][1]] for m in pair_matches])
            
            # RANSAC homography estimation
            best_H = None
            best_inliers = 0
            
            for _ in range(100):
                idx = np.random.choice(len(pts1), 4, replace=False)
                
                # Compute homography from 4 points
                H = self._compute_homography(pts1[idx], pts2[idx])
                
                if H is None:
                    continue
                
                # Count inliers
                pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
                projected = (H @ pts1_h.T).T
                projected = projected[:, :2] / projected[:, 2:3]
                
                errors = np.sqrt(np.sum((projected - pts2)**2, axis=1))
                inliers = np.sum(errors < 4.0)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_H = H
            
            if best_H is not None and j > 0:
                # Chain homographies
                homographies[j] = homographies[i] @ best_H
        
        return homographies
    
    def _compute_homography(self, src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
        """Compute homography from 4 point correspondences."""
        if len(src) != 4 or len(dst) != 4:
            return None
        
        A = np.zeros((8, 9))
        
        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            
            A[2*i] = [-x, -y, -1, 0, 0, 0, u*x, u*y, u]
            A[2*i+1] = [0, 0, 0, -x, -y, -1, v*x, v*y, v]
        
        try:
            _, _, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            return H / H[2, 2]
        except:
            return None
    
    def _warp_and_blend(
        self,
        images: List[np.ndarray],
        homographies: List[np.ndarray]
    ) -> np.ndarray:
        """Warp images and blend them."""
        # Find output size
        corners_list = []
        
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float32)
            
            H = homographies[i]
            corners_h = np.hstack([corners, np.ones((4, 1))])
            warped_corners = (H @ corners_h.T).T
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]
            corners_list.append(warped_corners)
        
        all_corners = np.vstack(corners_list)
        min_x, min_y = all_corners.min(axis=0).astype(int) - 10
        max_x, max_y = all_corners.max(axis=0).astype(int) + 10
        
        out_w = min(max_x - min_x, 10000)
        out_h = min(max_y - min_y, 10000)
        
        if out_w <= 0 or out_h <= 0:
            return images[0]
        
        # Create output
        result = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weights = np.zeros((out_h, out_w), dtype=np.float32)
        
        for i, img in enumerate(images):
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            
            H = homographies[i]
            
            # Inverse warp
            try:
                H_inv = np.linalg.inv(H)
            except:
                continue
            
            h, w = img.shape[:2]
            
            # Create coordinate grid
            y_coords, x_coords = np.meshgrid(
                np.arange(out_h) + min_y,
                np.arange(out_w) + min_x,
                indexing='ij'
            )
            
            coords = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=2)
            coords = coords.reshape(-1, 3)
            
            src_coords = (H_inv @ coords.T).T
            src_coords = src_coords[:, :2] / src_coords[:, 2:3]
            
            src_x = src_coords[:, 0].reshape(out_h, out_w)
            src_y = src_coords[:, 1].reshape(out_h, out_w)
            
            # Check valid pixels
            valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
            
            # Bilinear interpolation
            x0 = np.floor(src_x).astype(int)
            y0 = np.floor(src_y).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1
            
            x0 = np.clip(x0, 0, w - 1)
            x1 = np.clip(x1, 0, w - 1)
            y0 = np.clip(y0, 0, h - 1)
            y1 = np.clip(y1, 0, h - 1)
            
            dx = src_x - np.floor(src_x)
            dy = src_y - np.floor(src_y)
            
            for c in range(3):
                val = (img[y0, x0, c] * (1 - dx) * (1 - dy) +
                       img[y1, x0, c] * (1 - dx) * dy +
                       img[y0, x1, c] * dx * (1 - dy) +
                       img[y1, x1, c] * dx * dy)
                
                result[:, :, c] += val * valid
            
            weights += valid.astype(float)
        
        # Normalize
        weights = np.maximum(weights, 1)[:, :, np.newaxis]
        result = result / weights
        
        return result.astype(np.uint8)
    
    # Property getters/setters
    def registrationResol(self) -> float:
        return self._registration_resol
    
    def setRegistrationResol(self, resol: float) -> None:
        self._registration_resol = resol
    
    def seamEstimationResol(self) -> float:
        return self._seam_estimation_resol
    
    def setSeamEstimationResol(self, resol: float) -> None:
        self._seam_estimation_resol = resol
    
    def compositingResol(self) -> float:
        return self._compositing_resol
    
    def setCompositingResol(self, resol: float) -> None:
        self._compositing_resol = resol
    
    def panoConfidenceThresh(self) -> float:
        return self._pano_confidence_thresh
    
    def setPanoConfidenceThresh(self, thresh: float) -> None:
        self._pano_confidence_thresh = thresh


def createStitcher(mode: Stitcher.Mode = Stitcher.Mode.PANORAMA) -> Stitcher:
    """Create a Stitcher with given mode.
    
    Args:
        mode: Stitching mode (PANORAMA or SCANS)
    
    Returns:
        Stitcher instance
    """
    return Stitcher.create(mode)


__all__ = [
    "Stitcher",
    "createStitcher",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.