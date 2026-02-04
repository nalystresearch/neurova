# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Connected components analysis for Neurova.

Provides connectedComponents and connectedComponentsWithStats functions
for labeling connected regions in binary images.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Connectivity constants
CC_STAT_LEFT = 0
CC_STAT_TOP = 1
CC_STAT_WIDTH = 2
CC_STAT_HEIGHT = 3
CC_STAT_AREA = 4


def connectedComponents(
    image: np.ndarray,
    labels: Optional[np.ndarray] = None,
    connectivity: int = 8,
    ltype: int = 0  # CV_32S
) -> Tuple[int, np.ndarray]:
    """Label connected components in a binary image.
    
    Args:
        image: Binary image (8-bit single-channel)
        labels: Output label image (ignored, for compatibility)
        connectivity: 4 or 8 for pixel connectivity
        ltype: Output label type (ignored, always int32)
    
    Returns:
        Tuple of (num_labels, labels) where num_labels includes background (0)
    """
    img = np.asarray(image)
    
    if img.ndim != 2:
        raise ValueError("connectedComponents requires a 2D binary image")
    
    # Binarize if needed
    binary = (img > 0).astype(np.uint8)
    
    if HAS_SCIPY:
        # Use scipy's label function
        if connectivity == 4:
            structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:  # 8-connectivity
            structure = np.ones((3, 3), dtype=int)
        
        labeled, num_labels = ndimage.label(binary, structure=structure)
        return num_labels + 1, labeled.astype(np.int32)  # +1 to include background as label 0
    else:
        # Pure numpy two-pass algorithm
        labeled, num_labels = _connected_components_numpy(binary, connectivity)
        return num_labels + 1, labeled


def connectedComponentsWithStats(
    image: np.ndarray,
    labels: Optional[np.ndarray] = None,
    stats: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    connectivity: int = 8,
    ltype: int = 0
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Label connected components and compute statistics.
    
    Args:
        image: Binary image (8-bit single-channel)
        labels: Output label image (ignored)
        stats: Output statistics (ignored)
        centroids: Output centroids (ignored)
        connectivity: 4 or 8 for pixel connectivity
        ltype: Output label type (ignored)
    
    Returns:
        Tuple of (num_labels, labels, stats, centroids)
        - num_labels: Total number of labels (including background 0)
        - labels: Label image
        - stats: Statistics array (N x 5) with LEFT, TOP, WIDTH, HEIGHT, AREA
        - centroids: Centroid array (N x 2) with X, Y coordinates
    """
    num_labels, labeled = connectedComponents(image, connectivity=connectivity)
    
    # Compute statistics for each label
    stats_arr = np.zeros((num_labels, 5), dtype=np.int32)
    centroids_arr = np.zeros((num_labels, 2), dtype=np.float64)
    
    for label_id in range(num_labels):
        mask = labeled == label_id
        if not np.any(mask):
            continue
        
        # Find bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            continue
        
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        
        top = int(row_indices[0])
        bottom = int(row_indices[-1])
        left = int(col_indices[0])
        right = int(col_indices[-1])
        
        stats_arr[label_id, CC_STAT_LEFT] = left
        stats_arr[label_id, CC_STAT_TOP] = top
        stats_arr[label_id, CC_STAT_WIDTH] = right - left + 1
        stats_arr[label_id, CC_STAT_HEIGHT] = bottom - top + 1
        stats_arr[label_id, CC_STAT_AREA] = int(np.sum(mask))
        
        # Compute centroid
        y_indices, x_indices = np.where(mask)
        centroids_arr[label_id, 0] = np.mean(x_indices)  # X
        centroids_arr[label_id, 1] = np.mean(y_indices)  # Y
    
    return num_labels, labeled, stats_arr, centroids_arr


def _connected_components_numpy(binary: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    """Two-pass connected components algorithm using pure numpy."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    
    # Union-Find data structure
    parent = {}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    label_counter = 0
    
    # First pass
    for i in range(h):
        for j in range(w):
            if binary[i, j] == 0:
                continue
            
            # Get neighbor labels
            neighbors = []
            
            # 4-connectivity: left and top
            if j > 0 and labels[i, j-1] > 0:
                neighbors.append(labels[i, j-1])
            if i > 0 and labels[i-1, j] > 0:
                neighbors.append(labels[i-1, j])
            
            # 8-connectivity: add diagonals
            if connectivity == 8:
                if i > 0 and j > 0 and labels[i-1, j-1] > 0:
                    neighbors.append(labels[i-1, j-1])
                if i > 0 and j < w-1 and labels[i-1, j+1] > 0:
                    neighbors.append(labels[i-1, j+1])
            
            if not neighbors:
                label_counter += 1
                labels[i, j] = label_counter
                parent[label_counter] = label_counter
            else:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for n in neighbors:
                    union(n, min_label)
    
    # Second pass: resolve equivalences
    new_labels = {}
    new_label_counter = 0
    
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                root = find(labels[i, j])
                if root not in new_labels:
                    new_label_counter += 1
                    new_labels[root] = new_label_counter
                labels[i, j] = new_labels[root]
    
    return labels, new_label_counter


__all__ = [
    "connectedComponents",
    "connectedComponentsWithStats",
    "CC_STAT_LEFT",
    "CC_STAT_TOP",
    "CC_STAT_WIDTH",
    "CC_STAT_HEIGHT",
    "CC_STAT_AREA",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.