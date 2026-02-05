# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Advanced geometric transformations for image augmentation.

This module provides sophisticated geometric transformations including
elastic deformations, grid distortions, and morphological operations.
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from scipy import ndimage

__all__ = [
    # Elastic transformations
    'ElasticTransform', 'GridDistortion', 'OpticalDistortion',
    # Morphological operations
    'Erosion', 'Dilation', 'Opening', 'Closing',
    # Advanced geometric
    'PiecewiseAffine', 'RandomGridShuffle', 'CoarseDropout',
    # Motion blur
    'MotionBlur', 'ZoomBlur',
]


class ElasticTransform:
    """
    Apply elastic deformation to images.
    
    Useful for data augmentation in medical imaging and OCR tasks.
    
    Args:
        alpha: Intensity of the deformation
        sigma: Standard deviation of the Gaussian filter
        alpha_affine: Controls the strength of the affine component
        p: Probability of applying the transform
        
    References:
        Simard et al. (2003) "Best Practices for CNNs Applied to Visual Document Analysis"
    """
    
    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        alpha_affine: float = 0.0,
        p: float = 0.5
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        # Random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.rand(h, w) * 2 - 1),
            self.sigma
        ) * self.alpha
        dy = ndimage.gaussian_filter(
            (np.random.rand(h, w) * 2 - 1),
            self.sigma
        ) * self.alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        
        # Interpolate
        def _apply_transform(img_2d):
            return ndimage.map_coordinates(
                img_2d,
                [y_new.flatten(), x_new.flatten()],
                order=1,
                mode='reflect'
            ).reshape(h, w)
        
        if is_2d:
            return _apply_transform(image)
        elif is_chw:
            result = np.zeros_like(image)
            for i in range(c):
                result[i] = _apply_transform(image[i])
            return result
        else:
            result = np.zeros_like(image)
            for i in range(c):
                result[:, :, i] = _apply_transform(image[:, :, i])
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma})'


class GridDistortion:
    """
    Apply grid-based distortion to images.
    
    Divides the image into a grid and applies random displacements.
    
    Args:
        num_steps: Number of grid cells in each dimension
        distort_limit: Maximum distortion range
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: float = 0.3,
        p: float = 0.5
    ):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        # Grid points
        step_x = w / self.num_steps
        step_y = h / self.num_steps
        
        # Random displacements at grid points
        grid_x = np.arange(0, w + step_x, step_x)
        grid_y = np.arange(0, h + step_y, step_y)
        
        dx = np.random.uniform(-self.distort_limit, self.distort_limit, 
                               (len(grid_y), len(grid_x))) * step_x
        dy = np.random.uniform(-self.distort_limit, self.distort_limit,
                               (len(grid_y), len(grid_x))) * step_y
        
        # Interpolate displacements to full image
        from scipy.interpolate import RectBivariateSpline
        
        spline_dx = RectBivariateSpline(grid_y, grid_x, dx)
        spline_dy = RectBivariateSpline(grid_y, grid_x, dy)
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        dx_full = spline_dx(np.arange(h), np.arange(w))
        dy_full = spline_dy(np.arange(h), np.arange(w))
        
        x_new = np.clip(x + dx_full, 0, w - 1)
        y_new = np.clip(y + dy_full, 0, h - 1)
        
        def _apply_transform(img_2d):
            return ndimage.map_coordinates(
                img_2d,
                [y_new.flatten(), x_new.flatten()],
                order=1,
                mode='reflect'
            ).reshape(h, w)
        
        if is_2d:
            return _apply_transform(image)
        elif is_chw:
            result = np.zeros_like(image)
            for i in range(c):
                result[i] = _apply_transform(image[i])
            return result
        else:
            result = np.zeros_like(image)
            for i in range(c):
                result[:, :, i] = _apply_transform(image[:, :, i])
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_steps={self.num_steps}, distort_limit={self.distort_limit})'


class OpticalDistortion:
    """
    Apply barrel/pincushion distortion to images.
    
    Args:
        distort_limit: Distortion strength range
        shift_limit: Shift of the distortion center
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        distort_limit: float = 0.05,
        shift_limit: float = 0.05,
        p: float = 0.5
    ):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        # Random parameters
        k = np.random.uniform(-self.distort_limit, self.distort_limit)
        dx = np.random.uniform(-self.shift_limit, self.shift_limit)
        dy = np.random.uniform(-self.shift_limit, self.shift_limit)
        
        # Normalized coordinates
        cx, cy = w / 2 + dx * w, h / 2 + dy * h
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_norm = (x - cx) / max(w, h)
        y_norm = (y - cy) / max(w, h)
        
        r = np.sqrt(x_norm ** 2 + y_norm ** 2)
        
        # Radial distortion
        factor = 1 + k * r ** 2
        
        x_new = cx + (x - cx) * factor
        y_new = cy + (y - cy) * factor
        
        x_new = np.clip(x_new, 0, w - 1)
        y_new = np.clip(y_new, 0, h - 1)
        
        def _apply_transform(img_2d):
            return ndimage.map_coordinates(
                img_2d,
                [y_new.flatten(), x_new.flatten()],
                order=1,
                mode='reflect'
            ).reshape(h, w)
        
        if is_2d:
            return _apply_transform(image)
        elif is_chw:
            result = np.zeros_like(image)
            for i in range(c):
                result[i] = _apply_transform(image[i])
            return result
        else:
            result = np.zeros_like(image)
            for i in range(c):
                result[:, :, i] = _apply_transform(image[:, :, i])
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(distort_limit={self.distort_limit})'


class Erosion:
    """
    Apply morphological erosion.
    
    Args:
        kernel_size: Size of the structuring element
        iterations: Number of times to apply erosion
    """
    
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        structure = np.ones((self.kernel_size, self.kernel_size))
        
        if image.ndim == 2:
            return ndimage.grey_erosion(image, structure=structure)
        elif image.shape[0] <= 4:  # (C, H, W)
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = ndimage.grey_erosion(image[i], structure=structure)
            return result
        else:  # (H, W, C)
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.grey_erosion(image[:, :, i], structure=structure)
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'


class Dilation:
    """
    Apply morphological dilation.
    
    Args:
        kernel_size: Size of the structuring element
        iterations: Number of times to apply dilation
    """
    
    def __init__(self, kernel_size: int = 3, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        structure = np.ones((self.kernel_size, self.kernel_size))
        
        if image.ndim == 2:
            return ndimage.grey_dilation(image, structure=structure)
        elif image.shape[0] <= 4:  # (C, H, W)
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = ndimage.grey_dilation(image[i], structure=structure)
            return result
        else:  # (H, W, C)
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.grey_dilation(image[:, :, i], structure=structure)
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'


class Opening:
    """
    Apply morphological opening (erosion followed by dilation).
    
    Args:
        kernel_size: Size of the structuring element
    """
    
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        structure = np.ones((self.kernel_size, self.kernel_size))
        
        if image.ndim == 2:
            return ndimage.grey_opening(image, structure=structure)
        elif image.shape[0] <= 4:  # (C, H, W)
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = ndimage.grey_opening(image[i], structure=structure)
            return result
        else:  # (H, W, C)
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.grey_opening(image[:, :, i], structure=structure)
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'


class Closing:
    """
    Apply morphological closing (dilation followed by erosion).
    
    Args:
        kernel_size: Size of the structuring element
    """
    
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        structure = np.ones((self.kernel_size, self.kernel_size))
        
        if image.ndim == 2:
            return ndimage.grey_closing(image, structure=structure)
        elif image.shape[0] <= 4:  # (C, H, W)
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = ndimage.grey_closing(image[i], structure=structure)
            return result
        else:  # (H, W, C)
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.grey_closing(image[:, :, i], structure=structure)
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size})'


class PiecewiseAffine:
    """
    Apply piecewise affine transformations.
    
    Divides the image into regions and applies different affine
    transformations to each region.
    
    Args:
        scale: Maximum displacement for keypoints
        nb_rows: Number of rows in the grid
        nb_cols: Number of columns in the grid
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        scale: float = 0.05,
        nb_rows: int = 4,
        nb_cols: int = 4,
        p: float = 0.5
    ):
        self.scale = scale
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        # Create control points
        points_x = np.linspace(0, w, self.nb_cols + 1)
        points_y = np.linspace(0, h, self.nb_rows + 1)
        
        # Random displacements
        dx = np.random.uniform(-self.scale * w, self.scale * w, 
                               (self.nb_rows + 1, self.nb_cols + 1))
        dy = np.random.uniform(-self.scale * h, self.scale * h,
                               (self.nb_rows + 1, self.nb_cols + 1))
        
        # Keep borders fixed
        dx[0, :] = dx[-1, :] = 0
        dx[:, 0] = dx[:, -1] = 0
        dy[0, :] = dy[-1, :] = 0
        dy[:, 0] = dy[:, -1] = 0
        
        # Interpolate to full image
        from scipy.interpolate import RectBivariateSpline
        
        spline_dx = RectBivariateSpline(points_y, points_x, dx)
        spline_dy = RectBivariateSpline(points_y, points_x, dy)
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        dx_full = spline_dx(np.arange(h), np.arange(w))
        dy_full = spline_dy(np.arange(h), np.arange(w))
        
        x_new = np.clip(x + dx_full, 0, w - 1)
        y_new = np.clip(y + dy_full, 0, h - 1)
        
        def _apply_transform(img_2d):
            return ndimage.map_coordinates(
                img_2d,
                [y_new.flatten(), x_new.flatten()],
                order=1,
                mode='reflect'
            ).reshape(h, w)
        
        if is_2d:
            return _apply_transform(image)
        elif is_chw:
            result = np.zeros_like(image)
            for i in range(c):
                result[i] = _apply_transform(image[i])
            return result
        else:
            result = np.zeros_like(image)
            for i in range(c):
                result[:, :, i] = _apply_transform(image[:, :, i])
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(scale={self.scale}, nb_rows={self.nb_rows}, nb_cols={self.nb_cols})'


class RandomGridShuffle:
    """
    Divide the image into a grid and randomly shuffle the cells.
    
    Args:
        grid: Number of cells in (rows, cols)
        p: Probability of applying the transform
    """
    
    def __init__(self, grid: Tuple[int, int] = (3, 3), p: float = 0.5):
        self.grid = grid
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        n_rows, n_cols = self.grid
        cell_h, cell_w = h // n_rows, w // n_cols
        
        # Extract cells
        cells = []
        for i in range(n_rows):
            for j in range(n_cols):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                if is_2d:
                    cells.append(image[y1:y2, x1:x2].copy())
                elif is_chw:
                    cells.append(image[:, y1:y2, x1:x2].copy())
                else:
                    cells.append(image[y1:y2, x1:x2, :].copy())
        
        # Shuffle cells
        np.random.shuffle(cells)
        
        # Reconstruct image
        result = np.zeros_like(image)
        idx = 0
        for i in range(n_rows):
            for j in range(n_cols):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                if is_2d:
                    result[y1:y2, x1:x2] = cells[idx]
                elif is_chw:
                    result[:, y1:y2, x1:x2] = cells[idx]
                else:
                    result[y1:y2, x1:x2, :] = cells[idx]
                idx += 1
        
        return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(grid={self.grid})'


class CoarseDropout:
    """
    Drop rectangular regions from the image (Cutout).
    
    Args:
        max_holes: Maximum number of holes to drop
        max_height: Maximum height of holes (fraction of image height)
        max_width: Maximum width of holes (fraction of image width)
        min_holes: Minimum number of holes
        min_height: Minimum height of holes
        min_width: Minimum width of holes
        fill_value: Value to fill the holes with
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        max_holes: int = 8,
        max_height: float = 0.1,
        max_width: float = 0.1,
        min_holes: int = 1,
        min_height: float = 0.05,
        min_width: float = 0.05,
        fill_value: float = 0,
        p: float = 0.5
    ):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes
        self.min_height = min_height
        self.min_width = min_width
        self.fill_value = fill_value
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        result = image.copy()
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        num_holes = np.random.randint(self.min_holes, self.max_holes + 1)
        
        for _ in range(num_holes):
            hole_h = int(np.random.uniform(self.min_height, self.max_height) * h)
            hole_w = int(np.random.uniform(self.min_width, self.max_width) * w)
            
            y1 = np.random.randint(0, h - hole_h + 1)
            x1 = np.random.randint(0, w - hole_w + 1)
            y2, x2 = y1 + hole_h, x1 + hole_w
            
            if is_2d:
                result[y1:y2, x1:x2] = self.fill_value
            elif is_chw:
                result[:, y1:y2, x1:x2] = self.fill_value
            else:
                result[y1:y2, x1:x2, :] = self.fill_value
        
        return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(max_holes={self.max_holes})'


class MotionBlur:
    """
    Apply motion blur effect.
    
    Args:
        kernel_size: Size of the motion blur kernel
        angle: Angle of motion blur in degrees (or range)
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        kernel_size: int = 15,
        angle: Union[float, Tuple[float, float]] = (-45, 45),
        p: float = 0.5
    ):
        self.kernel_size = kernel_size
        if isinstance(angle, (int, float)):
            self.angle = (-angle, angle)
        else:
            self.angle = angle
        self.p = p
    
    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create a motion blur kernel."""
        kernel = np.zeros((size, size))
        
        # Line through center at given angle
        center = size // 2
        angle_rad = np.deg2rad(angle)
        
        for i in range(size):
            offset = i - center
            x = int(center + offset * np.cos(angle_rad))
            y = int(center + offset * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # Normalize
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        else:
            kernel[center, center] = 1
        
        return kernel
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        angle = np.random.uniform(*self.angle)
        kernel = self._create_motion_kernel(self.kernel_size, angle)
        
        if image.ndim == 2:
            return ndimage.convolve(image, kernel)
        elif image.shape[0] <= 4:  # (C, H, W)
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = ndimage.convolve(image[i], kernel)
            return result
        else:  # (H, W, C)
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = ndimage.convolve(image[:, :, i], kernel)
            return result
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, angle={self.angle})'


class ZoomBlur:
    """
    Apply zoom/radial blur effect.
    
    Args:
        max_factor: Maximum zoom factor
        step_factor: Step size for accumulating blur
        p: Probability of applying the transform
    """
    
    def __init__(
        self,
        max_factor: float = 0.1,
        step_factor: float = 0.01,
        p: float = 0.5
    ):
        self.max_factor = max_factor
        self.step_factor = step_factor
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return image
        
        from scipy.ndimage import zoom as scipy_zoom
        
        # Get image dimensions
        if image.ndim == 2:
            h, w = image.shape
            is_2d = True
        elif image.shape[0] <= 4:  # (C, H, W)
            c, h, w = image.shape
            is_2d = False
            is_chw = True
        else:  # (H, W, C)
            h, w, c = image.shape
            is_2d = False
            is_chw = False
        
        # Accumulate zoomed versions
        factor = np.random.uniform(0, self.max_factor)
        num_steps = max(1, int(factor / self.step_factor))
        
        result = image.astype(np.float64)
        
        for i in range(1, num_steps + 1):
            zoom_factor = 1 + i * self.step_factor
            
            if is_2d:
                zoomed = scipy_zoom(image, zoom_factor, order=1)
            elif is_chw:
                zoomed = scipy_zoom(image, (1, zoom_factor, zoom_factor), order=1)
            else:
                zoomed = scipy_zoom(image, (zoom_factor, zoom_factor, 1), order=1)
            
            # Center crop to original size
            if is_2d:
                zh, zw = zoomed.shape
                y1, x1 = (zh - h) // 2, (zw - w) // 2
                zoomed = zoomed[y1:y1+h, x1:x1+w]
            elif is_chw:
                zh, zw = zoomed.shape[1], zoomed.shape[2]
                y1, x1 = (zh - h) // 2, (zw - w) // 2
                zoomed = zoomed[:, y1:y1+h, x1:x1+w]
            else:
                zh, zw = zoomed.shape[0], zoomed.shape[1]
                y1, x1 = (zh - h) // 2, (zw - w) // 2
                zoomed = zoomed[y1:y1+h, x1:x1+w, :]
            
            result += zoomed
        
        result /= (num_steps + 1)
        
        return result.astype(image.dtype)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(max_factor={self.max_factor})'
