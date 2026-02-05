# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Watershed segmentation algorithm for Neurova."""

from __future__ import annotations
import numpy as np
from typing import Optional
from neurova.core.errors import ValidationError, ShapeError
from neurova.core.array_ops import ensure_2d


def distance_transform_edt(binary: np.ndarray) -> np.ndarray:
    """Euclidean distance transform of binary image.
    
    Args:
        binary: Binary image (True/1 for foreground, False/0 for background)
        
    Returns:
        Distance transform (float array)
    """
    binary = ensure_2d(binary).astype(bool)
    h, w = binary.shape
    dist = np.zeros((h, w), dtype=np.float64)
    
    # initialize: foreground pixels get 0, background gets infinity
    dist[~binary] = np.inf
    
    # two-pass algorithm (Felzenszwalb & Huttenlocher)
    # forward pass
    for i in range(h):
        for j in range(w):
            if binary[i, j]:
                min_dist = 0.0
                # check 4-connected neighbors
                if i > 0:
                    min_dist = max(min_dist, dist[i-1, j] + 1.0)
                if j > 0:
                    min_dist = max(min_dist, dist[i, j-1] + 1.0)
                dist[i, j] = min_dist if min_dist > 0 else 0.0
    
    # backward pass
    for i in range(h-1, -1, -1):
        for j in range(w-1, -1, -1):
            if binary[i, j]:
                min_dist = dist[i, j]
                # check 4-connected neighbors
                if i < h-1:
                    min_dist = max(min_dist, dist[i+1, j] + 1.0)
                if j < w-1:
                    min_dist = max(min_dist, dist[i, j+1] + 1.0)
                dist[i, j] = min_dist
    
    return dist


def watershed(
    image: np.ndarray,
    markers: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    compactness: float = 0.0,
) -> np.ndarray:
    """Watershed segmentation algorithm.
    
    Args:
        image: Input image (grayscale, will be treated as elevation map)
        markers: Marker array with labeled regions (int32), or None for automatic
        mask: Optional binary mask limiting the segmentation region
        compactness: Compactness parameter (0 = standard watershed, >0 = compact)
        
    Returns:
        Labeled segmentation (int32 array, 0 = boundaries)
    """
    image = ensure_2d(image).astype(np.float64)
    h, w = image.shape
    
    if markers is None:
        # auto-generate markers from local minima
        from neurova.morphology import binary_erode
        # simple approach: use distance transform peaks
        binary = image < np.percentile(image, 20)
        eroded = binary_erode(binary, iterations=3)
        markers = label_connected_components(eroded)
    else:
        markers = markers.astype(np.int32)
        if markers.shape != image.shape:
            raise ShapeError(image.shape, markers.shape, f"Markers shape {markers.shape} must match image shape {image.shape}")
    
    if mask is not None:
        mask = ensure_2d(mask).astype(bool)
        if mask.shape != image.shape:
            raise ShapeError(image.shape, mask.shape, f"Mask shape {mask.shape} must match image shape {image.shape}")
    else:
        mask = np.ones((h, w), dtype=bool)
    
    # initialize output
    labels = markers.copy()
    
    # priority queue simulation with sorted pixels
    # get all pixels sorted by intensity
    coords = np.column_stack(np.where(mask & (labels == 0)))
    if len(coords) == 0:
        return labels
    
    intensities = image[coords[:, 0], coords[:, 1]]
    sort_idx = np.argsort(intensities)
    coords = coords[sort_idx]
    
    # flood from markers
    for coord in coords:
        i, j = coord
        if labels[i, j] != 0:
            continue
            
        # check 8-connected neighbors
        neighbor_labels = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if labels[ni, nj] > 0:
                        neighbor_labels.append(labels[ni, nj])
        
        if neighbor_labels:
            # check if all neighbors have same label
            unique_labels = np.unique(neighbor_labels)
            if len(unique_labels) == 1:
                labels[i, j] = unique_labels[0]
            else:
                # watershed boundary - leave as 0
                labels[i, j] = 0
    
    return labels


def label_connected_components(
    binary: np.ndarray,
    connectivity: int = 8,
    background: int = 0,
) -> np.ndarray:
    """Label connected components in binary image.
    
    Args:
        binary: Binary image
        connectivity: 4 or 8 connectivity
        background: Value to treat as background (default: 0)
        
    Returns:
        Labeled image (int32, labels start from 1)
    """
    binary = ensure_2d(binary)
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    
    # union-find data structure
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # first pass: assign provisional labels
    for i in range(h):
        for j in range(w):
            if binary[i, j] == background:
                continue
                
            # check neighbors
            neighbors = []
            if i > 0 and labels[i-1, j] > 0:
                neighbors.append(labels[i-1, j])
            if j > 0 and labels[i, j-1] > 0:
                neighbors.append(labels[i, j-1])
            
            if connectivity == 8:
                if i > 0 and j > 0 and labels[i-1, j-1] > 0:
                    neighbors.append(labels[i-1, j-1])
                if i > 0 and j < w-1 and labels[i-1, j+1] > 0:
                    neighbors.append(labels[i-1, j+1])
            
            if not neighbors:
                labels[i, j] = current_label
                current_label += 1
            else:
                # assign minimum neighbor label
                min_label = min(neighbors)
                labels[i, j] = min_label
                # union all neighbor labels
                for n in neighbors:
                    union(min_label, n)
    
    # second pass: resolve equivalences
    label_map = {}
    next_label = 1
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                root = find(labels[i, j])
                if root not in label_map:
                    label_map[root] = next_label
                    next_label += 1
                labels[i, j] = label_map[root]
    
    return labels


def find_contours(
    binary: np.ndarray,
    level: float = 0.5,
) -> list[np.ndarray]:
    """Find contours in binary image using marching squares.
    
    Args:
        binary: Binary image
        level: Threshold level for contour extraction
        
    Returns:
        List of contours, each contour is Nx2 array of (row, col) coordinates
    """
    binary = ensure_2d(binary).astype(np.float64)
    h, w = binary.shape
    
    # pad image to handle boundaries
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    
    # track visited edges
    visited = np.zeros((h+1, w+1, 4), dtype=bool)  # 4 edges per cell
    contours = []
    
    # edge directions: 0=right, 1=up, 2=left, 3=down
    dx = [0, -1, 0, 1]
    dy = [1, 0, -1, 0]
    
    def trace_contour(start_i, start_j, start_dir):
        """Trace a single contour starting from given position and direction."""
        contour = []
        i, j, direction = start_i, start_j, start_dir
        
        while True:
            if visited[i, j, direction]:
                break
            visited[i, j, direction] = True
            
            # add point to contour
            if direction == 0:  # right edge
                contour.append([i + 0.5, j + 1])
            elif direction == 1:  # up edge
                contour.append([i, j + 0.5])
            elif direction == 2:  # left edge
                contour.append([i + 0.5, j])
            else:  # down edge
                contour.append([i + 1, j + 0.5])
            
            # move to next edge
            ni, nj = i + dx[direction], j + dy[direction]
            
            # determine next direction based on marching squares
            # check 2x2 cell configuration
            tl = padded[i, j] >= level
            tr = padded[i, j+1] >= level
            bl = padded[i+1, j] >= level
            br = padded[i+1, j+1] >= level
            
            case = (int(tl) << 3) | (int(tr) << 2) | (int(br) << 1) | int(bl)
            
            # simple next direction selection
            if case in [1, 3, 7]:
                direction = (direction + 3) % 4  # Turn left
            elif case in [2, 6, 14]:
                direction = (direction + 1) % 4  # Turn right
            elif case in [4, 12, 13]:
                direction = direction  # Straight
            else:
                break  # No contour or ambiguous
            
            i, j = ni, nj
            
            # check if we've returned to start
            if i == start_i and j == start_j and direction == start_dir:
                break
            
            # safety: limit contour length
            if len(contour) > (h + w) * 4:
                break
        
        return np.array(contour) if contour else None
    
    # find starting points for all contours
    for i in range(h):
        for j in range(w):
            # check all 4 edges of this cell
            for direction in range(4):
                if visited[i, j, direction]:
                    continue
                
                # check if this edge is a contour edge
                # this is a simplified check
                if direction == 0 and j < w-1:  # right edge
                    if (binary[i, j] >= level) != (binary[i, j+1] >= level):
                        contour = trace_contour(i, j, direction)
                        if contour is not None and len(contour) > 2:
                            contours.append(contour)
                elif direction == 1 and i > 0:  # up edge
                    if (binary[i, j] >= level) != (binary[i-1, j] >= level):
                        contour = trace_contour(i, j, direction)
                        if contour is not None and len(contour) > 2:
                            contours.append(contour)
    
    return contours


__all__ = [
    "distance_transform_edt",
    "watershed",
    "label_connected_components",
    "find_contours",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.