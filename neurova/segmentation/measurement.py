# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Region measurement and properties for Neurova."""

from __future__ import annotations
import numpy as np
from typing import Optional, NamedTuple
from neurova.core.errors import ValidationError
from neurova.core.array_ops import ensure_2d


class RegionProperties(NamedTuple):
    """Properties of a labeled region."""
    label: int
    area: int
    perimeter: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    major_axis_length: float
    minor_axis_length: float
    orientation: float
    eccentricity: float
    solidity: float
    extent: float


def regionprops(
    label_image: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
) -> list[RegionProperties]:
    """Measure properties of labeled image regions.
    
    Args:
        label_image: Labeled image (int array, 0 = background)
        intensity_image: Optional intensity image for intensity-based measurements
        
    Returns:
        List of RegionProperties for each labeled region
    """
    label_image = ensure_2d(label_image).astype(np.int32)
    h, w = label_image.shape
    
    if intensity_image is not None:
        intensity_image = ensure_2d(intensity_image)
        if intensity_image.shape != label_image.shape:
            raise ValidationError(
                "intensity_image",
                intensity_image.shape,
                f"shape matching label_image {label_image.shape}"
            )
    
    labels = np.unique(label_image)
    labels = labels[labels > 0]  # Exclude background
    
    properties = []
    for label in labels:
        mask = label_image == label
        props = _compute_region_props(mask, label)
        properties.append(props)
    
    return properties


def _compute_region_props(mask: np.ndarray, label: int) -> RegionProperties:
    """Compute properties for a single binary region."""
    h, w = mask.shape
    coords = np.column_stack(np.where(mask))
    
    if len(coords) == 0:
        # empty region
        return RegionProperties(
            label=label,
            area=0,
            perimeter=0.0,
            centroid=(0.0, 0.0),
            bbox=(0, 0, 0, 0),
            major_axis_length=0.0,
            minor_axis_length=0.0,
            orientation=0.0,
            eccentricity=0.0,
            solidity=0.0,
            extent=0.0,
        )
    
    # area
    area = len(coords)
    
    # bounding box
    min_row, min_col = coords.min(axis=0)
    max_row, max_col = coords.max(axis=0)
    bbox = (int(min_row), int(min_col), int(max_row), int(max_col))
    
    # centroid
    centroid_row = coords[:, 0].mean()
    centroid_col = coords[:, 1].mean()
    centroid = (float(centroid_row), float(centroid_col))
    
    # perimeter (count edge pixels with 4-connectivity)
    perimeter = 0.0
    for r, c in coords:
        # check 4-connected neighbors
        neighbors = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                neighbors += 1
        # edge pixel if has fewer than 4 neighbors
        if neighbors < 4:
            perimeter += (4 - neighbors) * 0.25
    perimeter = max(perimeter, 1.0)
    
    # moments for orientation and axis lengths
    mu00 = area
    mu10 = coords[:, 0].sum()
    mu01 = coords[:, 1].sum()
    
    # central moments
    mu20 = ((coords[:, 0] - centroid_row) ** 2).sum()
    mu02 = ((coords[:, 1] - centroid_col) ** 2).sum()
    mu11 = ((coords[:, 0] - centroid_row) * (coords[:, 1] - centroid_col)).sum()
    
    # normalize
    mu20 /= mu00
    mu02 /= mu00
    mu11 /= mu00
    
    # eigenvalues of covariance matrix for axis lengths
    # covariance matrix: [[mu20, mu11], [mu11, mu02]]
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11 * mu11
    
    # eigenvalues
    discriminant = max(trace * trace - 4 * det, 0.0)
    lambda1 = (trace + np.sqrt(discriminant)) / 2.0
    lambda2 = (trace - np.sqrt(discriminant)) / 2.0
    
    # axis lengths (4 * sqrt(eigenvalue) for 95% coverage)
    major_axis_length = 4.0 * np.sqrt(max(lambda1, 1e-10))
    minor_axis_length = 4.0 * np.sqrt(max(lambda2, 1e-10))
    
    # orientation (angle of major axis)
    if mu20 - mu02 == 0:
        orientation = np.pi / 4 if mu11 > 0 else -np.pi / 4
    else:
        orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    
    # eccentricity
    if minor_axis_length > 0:
        eccentricity = np.sqrt(1.0 - (minor_axis_length / major_axis_length) ** 2)
    else:
        eccentricity = 1.0
    
    # solidity (area / convex hull area, approximated as area / bbox area)
    bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
    solidity = area / bbox_area if bbox_area > 0 else 1.0
    
    # extent (same as solidity for now)
    extent = solidity
    
    return RegionProperties(
        label=label,
        area=area,
        perimeter=perimeter,
        centroid=centroid,
        bbox=bbox,
        major_axis_length=major_axis_length,
        minor_axis_length=minor_axis_length,
        orientation=orientation,
        eccentricity=eccentricity,
        solidity=solidity,
        extent=extent,
    )


def label_stats(label_image: np.ndarray) -> dict[int, dict]:
    """Compute basic statistics for each labeled region.
    
    Args:
        label_image: Labeled image (int array, 0 = background)
        
    Returns:
        Dictionary mapping label -> stats dict with keys:
        - area: number of pixels
        - centroid: (row, col) tuple
        - bbox: (min_row, min_col, max_row, max_col) tuple
    """
    label_image = ensure_2d(label_image).astype(np.int32)
    labels = np.unique(label_image)
    labels = labels[labels > 0]
    
    stats = {}
    for label in labels:
        mask = label_image == label
        coords = np.column_stack(np.where(mask))
        
        if len(coords) == 0:
            stats[int(label)] = {
                "area": 0,
                "centroid": (0.0, 0.0),
                "bbox": (0, 0, 0, 0),
            }
        else:
            stats[int(label)] = {
                "area": len(coords),
                "centroid": (float(coords[:, 0].mean()), float(coords[:, 1].mean())),
                "bbox": (
                    int(coords[:, 0].min()),
                    int(coords[:, 1].min()),
                    int(coords[:, 0].max()),
                    int(coords[:, 1].max()),
                ),
            }
    
    return stats


__all__ = [
    "RegionProperties",
    "regionprops",
    "label_stats",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.