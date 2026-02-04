# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
neurova.imgproc.contours - Contour detection and analysis functions

Provides Neurova contour operations.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Union
import numpy as np

# Contour Retrieval Modes

RETR_EXTERNAL = 0     # Retrieve only extreme outer contours
RETR_LIST = 1         # Retrieve all contours without hierarchy
RETR_CCOMP = 2        # Retrieve all contours in 2-level hierarchy
RETR_TREE = 3         # Retrieve all contours in full hierarchy
RETR_FLOODFILL = 4    # Flood fill based retrieval

# Contour Approximation Methods

CHAIN_APPROX_NONE = 1        # Store all contour points
CHAIN_APPROX_SIMPLE = 2      # Compress horizontal, vertical, diagonal segments
CHAIN_APPROX_TC89_L1 = 3     # Teh-Chin L1 approximation
CHAIN_APPROX_TC89_KCOS = 4   # Teh-Chin KCOS approximation


def findContours(
    image: np.ndarray,
    mode: int = RETR_LIST,
    method: int = CHAIN_APPROX_SIMPLE,
    contours: Optional[List] = None,
    hierarchy: Optional[np.ndarray] = None,
    offset: Tuple[int, int] = (0, 0)
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Find contours in a binary image.
    
    Args:
        image: 8-bit single-channel binary image
        mode: Contour retrieval mode (RETR_*)
        method: Contour approximation method (CHAIN_APPROX_*)
        contours: Optional output list (modified in place)
        hierarchy: Optional output hierarchy array
        offset: Optional offset for contour points
    
    Returns:
        Tuple of (contours list, hierarchy array)
    """
    if image.ndim != 2:
        raise ValueError("Input image must be single-channel (grayscale/binary)")
    
    # Ensure binary image
    binary = (image > 0).astype(np.uint8)
    
    # Use border-following algorithm (Suzuki-Abe)
    result_contours = []
    result_hierarchy = []
    
    # Create padded image for border following
    padded = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = binary
    
    # Label for contours
    nbd = 1  # Current border number
    lnbd = 1  # Last border number
    
    parent_stack = [-1]  # Parent hierarchy stack
    
    for i in range(1, padded.shape[0] - 1):
        lnbd = 1
        for j in range(1, padded.shape[1] - 1):
            # Check if we're at a border pixel
            fij = padded[i, j]
            
            if fij == 0:
                continue
            
            is_outer = (padded[i, j - 1] == 0 and fij == 1)
            is_hole = (fij >= 1 and padded[i, j + 1] == 0)
            
            if is_outer or is_hole:
                nbd += 1
                
                if is_outer:
                    i2, j2 = i, j - 1
                    parent = lnbd if lnbd > 1 else -1
                else:
                    i2, j2 = i, j + 1
                    if fij > 1:
                        lnbd = fij
                    parent = lnbd if lnbd > 1 else -1
                
                # Trace contour starting from (i, j) in direction (i2, j2)
                contour = _trace_contour(padded, i, j, i2, j2, nbd)
                
                if contour is not None and len(contour) >= 3:
                    # Convert to original coordinates and apply offset
                    contour_array = np.array(contour, dtype=np.int32)
                    contour_array[:, 0] -= 1  # Remove padding offset
                    contour_array[:, 1] -= 1
                    contour_array[:, 0] += offset[0]
                    contour_array[:, 1] += offset[1]
                    
                    # Swap x, y for Neurova format [N, 1, 2]
                    contour_array = contour_array[:, ::-1]
                    contour_array = contour_array.reshape(-1, 1, 2)
                    
                    result_contours.append(contour_array)
                    
                    # Build hierarchy: [next, prev, child, parent]
                    result_hierarchy.append([
                        -1,  # Next contour at same level
                        -1,  # Previous contour at same level
                        -1,  # First child
                        parent - 2 if parent > 1 else -1  # Parent
                    ])
            
            if fij != 0 and fij != 1:
                lnbd = abs(fij)
    
    # Update hierarchy relationships
    if result_hierarchy:
        hierarchy_array = np.array(result_hierarchy, dtype=np.int32)
        hierarchy_array = hierarchy_array.reshape(1, -1, 4)
    else:
        hierarchy_array = np.zeros((1, 0, 4), dtype=np.int32)
    
    # Filter based on mode
    if mode == RETR_EXTERNAL:
        # Keep only contours with no parent
        filtered = []
        for idx, (c, h) in enumerate(zip(result_contours, result_hierarchy)):
            if h[3] == -1:  # No parent
                filtered.append(c)
        result_contours = filtered
        hierarchy_array = np.zeros((1, len(filtered), 4), dtype=np.int32)
    
    # Apply approximation method
    if method == CHAIN_APPROX_SIMPLE:
        result_contours = [_compress_contour(c) for c in result_contours]
    
    return result_contours, hierarchy_array


def _trace_contour(
    image: np.ndarray, 
    i: int, 
    j: int, 
    i2: int, 
    j2: int,
    nbd: int
) -> Optional[List[Tuple[int, int]]]:
    """Trace a contour using border following algorithm."""
    # 8-connectivity neighbors (clockwise from right)
    neighbors = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    
    # Find starting direction
    di, dj = i2 - i, j2 - j
    try:
        start_dir = neighbors.index((di, dj))
    except ValueError:
        return None
    
    contour = [(i, j)]
    ci, cj = i, j
    
    # Find first non-zero neighbor (clockwise)
    direction = (start_dir + 1) % 8
    found = False
    
    for _ in range(8):
        ni, nj = ci + neighbors[direction][0], cj + neighbors[direction][1]
        if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
            if image[ni, nj] != 0:
                found = True
                break
        direction = (direction + 1) % 8
    
    if not found:
        # Isolated point
        return contour
    
    # Trace the contour
    start_i, start_j = i, j
    prev_i, prev_j = ci, cj
    ci, cj = ni, nj
    
    max_iterations = image.shape[0] * image.shape[1]
    
    for iteration in range(max_iterations):
        contour.append((ci, cj))
        
        # Mark pixel
        if image[ci, cj] == 1:
            image[ci, cj] = nbd
        elif image[ci, cj] != 0 and image[ci, cj] != nbd:
            image[ci, cj] = -nbd
        
        # Find direction we came from
        di, dj = prev_i - ci, prev_j - cj
        try:
            from_dir = neighbors.index((di, dj))
        except ValueError:
            break
        
        # Search for next pixel (clockwise from opposite direction)
        direction = (from_dir + 1) % 8
        found = False
        
        for _ in range(8):
            ni, nj = ci + neighbors[direction][0], cj + neighbors[direction][1]
            if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                if image[ni, nj] != 0:
                    found = True
                    break
            direction = (direction + 1) % 8
        
        if not found:
            break
        
        prev_i, prev_j = ci, cj
        ci, cj = ni, nj
        
        # Check if we're back at start
        if ci == start_i and cj == start_j:
            break
    
    return contour


def _compress_contour(contour: np.ndarray) -> np.ndarray:
    """Compress contour by removing redundant points on straight lines."""
    if len(contour) <= 2:
        return contour
    
    points = contour.reshape(-1, 2)
    compressed = [points[0]]
    
    for i in range(1, len(points) - 1):
        # Check if current point is on line between prev and next
        p0, p1, p2 = points[i - 1], points[i], points[i + 1]
        
        dx1, dy1 = p1[0] - p0[0], p1[1] - p0[1]
        dx2, dy2 = p2[0] - p1[0], p2[1] - p1[1]
        
        # Not collinear if direction changes
        if (dx1 != dx2) or (dy1 != dy2):
            compressed.append(p1)
    
    compressed.append(points[-1])
    
    return np.array(compressed, dtype=np.int32).reshape(-1, 1, 2)


def contourArea(contour: np.ndarray, oriented: bool = False) -> float:
    """Calculate the contour area using the shoelace formula.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array)
        oriented: If True, return signed area (positive for counter-clockwise)
    
    Returns:
        Area of the contour
    """
    if contour.size == 0:
        return 0.0
    
    # Flatten to Nx2
    points = contour.reshape(-1, 2).astype(np.float64)
    
    if len(points) < 3:
        return 0.0
    
    # Shoelace formula
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 
                        x[-1] * y[0] - x[0] * y[-1])
    
    if oriented:
        # Calculate signed area
        signed_area = 0.5 * (np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 
                            x[-1] * y[0] - x[0] * y[-1])
        return signed_area
    
    return area


def arcLength(curve: np.ndarray, closed: bool = False) -> float:
    """Calculate the perimeter (arc length) of a contour/curve.
    
    Args:
        curve: Input curve (Nx1x2 or Nx2 array)
        closed: Whether the curve is closed
    
    Returns:
        Arc length of the curve
    """
    if curve.size == 0:
        return 0.0
    
    points = curve.reshape(-1, 2).astype(np.float64)
    
    if len(points) < 2:
        return 0.0
    
    # Calculate distances between consecutive points
    diffs = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    length = np.sum(distances)
    
    # Add closing segment if closed
    if closed and len(points) >= 2:
        closing_dist = np.sqrt(np.sum((points[-1] - points[0]) ** 2))
        length += closing_dist
    
    return length


def boundingRect(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate the up-right bounding rectangle of a contour.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array)
    
    Returns:
        Tuple of (x, y, width, height)
    """
    if contour.size == 0:
        return (0, 0, 0, 0)
    
    points = contour.reshape(-1, 2)
    
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def minAreaRect(contour: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Find a rotated rectangle of minimum area enclosing a contour.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array)
    
    Returns:
        Tuple of ((center_x, center_y), (width, height), angle)
    """
    if contour.size == 0:
        return ((0.0, 0.0), (0.0, 0.0), 0.0)
    
    points = contour.reshape(-1, 2).astype(np.float64)
    
    if len(points) < 3:
        if len(points) == 1:
            return ((points[0, 0], points[0, 1]), (0.0, 0.0), 0.0)
        elif len(points) == 2:
            center = ((points[0, 0] + points[1, 0]) / 2, 
                     (points[0, 1] + points[1, 1]) / 2)
            length = np.sqrt(np.sum((points[1] - points[0]) ** 2))
            angle = np.degrees(np.arctan2(points[1, 1] - points[0, 1],
                                         points[1, 0] - points[0, 0]))
            return (center, (length, 0.0), angle)
    
    # Get convex hull first for efficiency
    hull = convexHull(contour)
    hull_points = hull.reshape(-1, 2)
    
    if len(hull_points) < 3:
        return boundingRect(contour)[:2], boundingRect(contour)[2:], 0.0
    
    # Rotating calipers algorithm (simplified)
    min_area = float('inf')
    best_rect = None
    
    n = len(hull_points)
    for i in range(n):
        # Edge vector
        edge = hull_points[(i + 1) % n] - hull_points[i]
        edge_length = np.linalg.norm(edge)
        if edge_length < 1e-10:
            continue
        
        # Unit vector along edge
        unit_edge = edge / edge_length
        
        # Perpendicular vector
        unit_perp = np.array([-unit_edge[1], unit_edge[0]])
        
        # Project all points onto edge and perpendicular
        centered = hull_points - hull_points[i]
        proj_edge = centered @ unit_edge
        proj_perp = centered @ unit_perp
        
        min_edge, max_edge = proj_edge.min(), proj_edge.max()
        min_perp, max_perp = proj_perp.min(), proj_perp.max()
        
        width = max_edge - min_edge
        height = max_perp - min_perp
        area = width * height
        
        if area < min_area:
            min_area = area
            
            # Calculate center in original coordinates
            center_proj = np.array([(min_edge + max_edge) / 2, (min_perp + max_perp) / 2])
            center = hull_points[i] + center_proj[0] * unit_edge + center_proj[1] * unit_perp
            
            # Angle of the edge
            angle = np.degrees(np.arctan2(unit_edge[1], unit_edge[0]))
            
            best_rect = ((float(center[0]), float(center[1])), 
                        (float(width), float(height)), 
                        float(angle))
    
    return best_rect if best_rect else ((0.0, 0.0), (0.0, 0.0), 0.0)


def minEnclosingCircle(contour: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Find the minimum enclosing circle of a contour.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array)
    
    Returns:
        Tuple of ((center_x, center_y), radius)
    """
    if contour.size == 0:
        return ((0.0, 0.0), 0.0)
    
    points = contour.reshape(-1, 2).astype(np.float64)
    
    if len(points) == 1:
        return ((float(points[0, 0]), float(points[0, 1])), 0.0)
    
    if len(points) == 2:
        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[1] - points[0]) / 2
        return ((float(center[0]), float(center[1])), float(radius))
    
    # Welzl's algorithm (randomized, but we use deterministic for simplicity)
    return _welzl_circle(points.tolist(), [])


def _welzl_circle(P: List, R: List) -> Tuple[Tuple[float, float], float]:
    """Welzl's algorithm for minimum enclosing circle."""
    if len(P) == 0 or len(R) == 3:
        return _make_circle(R)
    
    p = P.pop()
    center, radius = _welzl_circle(P.copy(), R.copy())
    
    # Check if p is inside current circle
    if center is not None:
        dist = np.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
        if dist <= radius + 1e-10:
            P.append(p)
            return center, radius
    
    # p must be on boundary
    result = _welzl_circle(P, R + [p])
    P.append(p)
    return result


def _make_circle(boundary: List) -> Tuple[Tuple[float, float], float]:
    """Create circle from boundary points."""
    if len(boundary) == 0:
        return ((0.0, 0.0), 0.0)
    elif len(boundary) == 1:
        return ((float(boundary[0][0]), float(boundary[0][1])), 0.0)
    elif len(boundary) == 2:
        p1, p2 = boundary
        center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        radius = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) / 2
        return (center, radius)
    else:
        # Three points - circumscribed circle
        p1, p2, p3 = boundary[:3]
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            # Collinear - use two furthest points
            d12 = (ax - bx) ** 2 + (ay - by) ** 2
            d23 = (bx - cx) ** 2 + (by - cy) ** 2
            d13 = (ax - cx) ** 2 + (ay - cy) ** 2
            if d12 >= d23 and d12 >= d13:
                return _make_circle([p1, p2])
            elif d23 >= d13:
                return _make_circle([p2, p3])
            else:
                return _make_circle([p1, p3])
        
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + 
              (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + 
              (cx * cx + cy * cy) * (bx - ax)) / d
        
        radius = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
        return ((ux, uy), radius)


def convexHull(points: np.ndarray, clockwise: bool = False, returnPoints: bool = True) -> np.ndarray:
    """Find the convex hull of a point set.
    
    Args:
        points: Input points (Nx1x2 or Nx2 array)
        clockwise: If True, output hull is clockwise oriented
        returnPoints: If True, return hull points; else return indices
    
    Returns:
        Convex hull as Nx1x2 array (if returnPoints) or Nx1 indices
    """
    if points.size == 0:
        return np.zeros((0, 1, 2), dtype=np.int32)
    
    pts = points.reshape(-1, 2)
    
    if len(pts) < 3:
        if returnPoints:
            return pts.reshape(-1, 1, 2).astype(np.int32)
        return np.arange(len(pts)).reshape(-1, 1).astype(np.int32)
    
    # Graham scan algorithm
    # Find bottom-most point (or left-most in case of tie)
    start_idx = 0
    for i in range(1, len(pts)):
        if pts[i, 1] < pts[start_idx, 1]:
            start_idx = i
        elif pts[i, 1] == pts[start_idx, 1] and pts[i, 0] < pts[start_idx, 0]:
            start_idx = i
    
    start = pts[start_idx]
    
    # Sort points by polar angle
    def polar_angle(p):
        return np.arctan2(p[1] - start[1], p[0] - start[0])
    
    indices = list(range(len(pts)))
    indices.sort(key=lambda i: (polar_angle(pts[i]), 
                                np.sum((pts[i] - start) ** 2)))
    
    # Build hull
    hull_indices = []
    
    def ccw(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    for idx in indices:
        while len(hull_indices) > 1 and ccw(pts[hull_indices[-2]], pts[hull_indices[-1]], pts[idx]) <= 0:
            hull_indices.pop()
        hull_indices.append(idx)
    
    if clockwise:
        hull_indices = hull_indices[::-1]
    
    if returnPoints:
        hull_points = pts[hull_indices]
        return hull_points.reshape(-1, 1, 2).astype(np.int32)
    else:
        return np.array(hull_indices).reshape(-1, 1).astype(np.int32)


def approxPolyDP(curve: np.ndarray, epsilon: float, closed: bool = False) -> np.ndarray:
    """Approximate a polygonal curve with specified precision.
    
    Uses the Ramer-Douglas-Peucker algorithm.
    
    Args:
        curve: Input curve (Nx1x2 or Nx2 array)
        epsilon: Approximation accuracy (max distance)
        closed: Whether the curve is closed
    
    Returns:
        Approximated curve as Nx1x2 array
    """
    if curve.size == 0:
        return np.zeros((0, 1, 2), dtype=np.int32)
    
    points = curve.reshape(-1, 2).astype(np.float64)
    
    if len(points) <= 2:
        return points.reshape(-1, 1, 2).astype(np.int32)
    
    if closed:
        # For closed curves, find the point with max distance from line between neighbors
        points_list = points.tolist()
        result = _rdp_closed(points_list, epsilon)
    else:
        result = _rdp(points.tolist(), epsilon)
    
    return np.array(result, dtype=np.int32).reshape(-1, 1, 2)


def _rdp(points: List, epsilon: float) -> List:
    """Ramer-Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points
    
    # Find point with maximum distance from line
    start, end = np.array(points[0]), np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        # All points at same location
        return [points[0], points[-1]]
    
    line_unit = line_vec / line_len
    
    max_dist = 0
    max_idx = 0
    
    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        proj = np.dot(pt - start, line_unit)
        
        if proj <= 0:
            dist = np.linalg.norm(pt - start)
        elif proj >= line_len:
            dist = np.linalg.norm(pt - end)
        else:
            closest = start + proj * line_unit
            dist = np.linalg.norm(pt - closest)
        
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > epsilon:
        # Recursively simplify
        left = _rdp(points[:max_idx + 1], epsilon)
        right = _rdp(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _rdp_closed(points: List, epsilon: float) -> List:
    """RDP for closed curves."""
    if len(points) <= 3:
        return points
    
    # Find point with max distance from line connecting its neighbors
    max_dist = 0
    max_idx = 0
    n = len(points)
    
    for i in range(n):
        p = np.array(points[i])
        p_prev = np.array(points[(i - 1) % n])
        p_next = np.array(points[(i + 1) % n])
        
        line_vec = p_next - p_prev
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-10:
            continue
        
        dist = abs(np.cross(line_vec, p - p_prev)) / line_len
        
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist <= epsilon:
        return points
    
    # Split at max_idx and recurse
    result = _rdp(points, epsilon)
    return result


def moments(contour: np.ndarray, binaryImage: bool = False) -> dict:
    """Calculate all moments of a contour up to third order.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array) or image
        binaryImage: If True, treat non-zero pixels as 1
    
    Returns:
        Dictionary with moment values (m00, m10, m01, m20, m11, m02, m30, m21, m12, m03,
        mu20, mu11, mu02, mu30, mu21, mu12, mu03, nu20, nu11, nu02, nu30, nu21, nu12, nu03)
    """
    result = {
        'm00': 0.0, 'm10': 0.0, 'm01': 0.0,
        'm20': 0.0, 'm11': 0.0, 'm02': 0.0,
        'm30': 0.0, 'm21': 0.0, 'm12': 0.0, 'm03': 0.0,
        'mu20': 0.0, 'mu11': 0.0, 'mu02': 0.0,
        'mu30': 0.0, 'mu21': 0.0, 'mu12': 0.0, 'mu03': 0.0,
        'nu20': 0.0, 'nu11': 0.0, 'nu02': 0.0,
        'nu30': 0.0, 'nu21': 0.0, 'nu12': 0.0, 'nu03': 0.0
    }
    
    if contour.size == 0:
        return result
    
    # Check if input is an image or contour
    if contour.ndim == 2 and contour.shape[1] > 2:
        # Input is an image
        image = contour
        if binaryImage:
            image = (image > 0).astype(np.float64)
        else:
            image = image.astype(np.float64)
        
        y, x = np.mgrid[:image.shape[0], :image.shape[1]]
        
        result['m00'] = np.sum(image)
        result['m10'] = np.sum(x * image)
        result['m01'] = np.sum(y * image)
        result['m20'] = np.sum(x * x * image)
        result['m11'] = np.sum(x * y * image)
        result['m02'] = np.sum(y * y * image)
        result['m30'] = np.sum(x * x * x * image)
        result['m21'] = np.sum(x * x * y * image)
        result['m12'] = np.sum(x * y * y * image)
        result['m03'] = np.sum(y * y * y * image)
    else:
        # Input is a contour
        points = contour.reshape(-1, 2).astype(np.float64)
        
        if len(points) < 3:
            return result
        
        # Use Green's theorem for contour moments
        n = len(points)
        for i in range(n):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % n]
            
            a = x0 * y1 - x1 * y0
            
            result['m00'] += a
            result['m10'] += a * (x0 + x1)
            result['m01'] += a * (y0 + y1)
            result['m20'] += a * (x0 * x0 + x0 * x1 + x1 * x1)
            result['m11'] += a * (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1)
            result['m02'] += a * (y0 * y0 + y0 * y1 + y1 * y1)
        
        result['m00'] /= 2
        result['m10'] /= 6
        result['m01'] /= 6
        result['m20'] /= 12
        result['m11'] /= 24
        result['m02'] /= 12
    
    # Central moments
    if result['m00'] != 0:
        cx = result['m10'] / result['m00']
        cy = result['m01'] / result['m00']
        
        result['mu20'] = result['m20'] - cx * result['m10']
        result['mu11'] = result['m11'] - cx * result['m01']
        result['mu02'] = result['m02'] - cy * result['m01']
        result['mu30'] = result['m30'] - 3 * cx * result['m20'] + 2 * cx * cx * result['m10']
        result['mu21'] = result['m21'] - 2 * cx * result['m11'] - cy * result['m20'] + 2 * cx * cx * result['m01']
        result['mu12'] = result['m12'] - 2 * cy * result['m11'] - cx * result['m02'] + 2 * cy * cy * result['m10']
        result['mu03'] = result['m03'] - 3 * cy * result['m02'] + 2 * cy * cy * result['m01']
        
        # Normalized central moments
        m00 = result['m00']
        result['nu20'] = result['mu20'] / (m00 ** 1.5) if m00 > 0 else 0
        result['nu11'] = result['mu11'] / (m00 ** 1.5) if m00 > 0 else 0
        result['nu02'] = result['mu02'] / (m00 ** 1.5) if m00 > 0 else 0
        result['nu30'] = result['mu30'] / (m00 ** 2) if m00 > 0 else 0
        result['nu21'] = result['mu21'] / (m00 ** 2) if m00 > 0 else 0
        result['nu12'] = result['mu12'] / (m00 ** 2) if m00 > 0 else 0
        result['nu03'] = result['mu03'] / (m00 ** 2) if m00 > 0 else 0
    
    return result


def isContourConvex(contour: np.ndarray) -> bool:
    """Test whether a contour is convex.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array)
    
    Returns:
        True if convex, False otherwise
    """
    if contour.size == 0:
        return False
    
    points = contour.reshape(-1, 2)
    
    if len(points) < 3:
        return True
    
    n = len(points)
    sign = None
    
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        
        if cross != 0:
            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False
    
    return True


def pointPolygonTest(contour: np.ndarray, pt: Tuple[float, float], measureDist: bool = False) -> float:
    """Test if a point is inside, outside, or on a contour.
    
    Args:
        contour: Input contour (Nx1x2 or Nx2 array)
        pt: Point to test (x, y)
        measureDist: If True, return signed distance; else return +1/0/-1
    
    Returns:
        If measureDist: signed distance (positive inside, negative outside)
        Else: +1 (inside), 0 (on edge), -1 (outside)
    """
    if contour.size == 0:
        return -1.0 if not measureDist else float('-inf')
    
    points = contour.reshape(-1, 2).astype(np.float64)
    px, py = float(pt[0]), float(pt[1])
    
    n = len(points)
    if n < 3:
        return -1.0 if not measureDist else float('-inf')
    
    # Check if point is on any edge
    min_dist = float('inf')
    
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        
        # Distance from point to line segment
        dx, dy = x2 - x1, y2 - y1
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy + 1e-10)))
        
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        dist = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
        
        min_dist = min(min_dist, dist)
        
        # Check if point is on edge
        if dist < 1e-10:
            return 0.0 if not measureDist else 0.0
    
    # Ray casting to determine inside/outside
    inside = False
    
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-10) + x1):
            inside = not inside
    
    if measureDist:
        return min_dist if inside else -min_dist
    else:
        return 1.0 if inside else -1.0


def convexityDefects(
    contour: np.ndarray,
    convexhull: np.ndarray
) -> np.ndarray:
    """Find convexity defects of a contour.
    
    Args:
        contour: Input contour
        convexhull: Convex hull obtained from convexHull with returnPoints=False
    
    Returns:
        Array of defects (Nx1x4): [start_idx, end_idx, farthest_pt_idx, fixpt_depth]
    """
    contour = np.asarray(contour)
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    
    hull = np.asarray(convexhull).flatten()
    
    if len(hull) < 3:
        return np.zeros((0, 1, 4), dtype=np.int32)
    
    defects = []
    n_hull = len(hull)
    
    for i in range(n_hull):
        start_idx = hull[i]
        end_idx = hull[(i + 1) % n_hull]
        
        # Get hull edge
        start_pt = contour[start_idx]
        end_pt = contour[end_idx]
        
        # Find farthest point between start and end on contour
        max_dist = 0
        farthest_idx = start_idx
        
        # Walk along contour from start to end
        if start_idx <= end_idx:
            indices = range(start_idx, end_idx + 1)
        else:
            indices = list(range(start_idx, len(contour))) + list(range(0, end_idx + 1))
        
        for j in indices:
            pt = contour[j]
            
            # Distance from point to line
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            
            line_len = np.sqrt(dx * dx + dy * dy)
            if line_len < 1e-10:
                dist = np.sqrt((pt[0] - start_pt[0])**2 + (pt[1] - start_pt[1])**2)
            else:
                dist = abs((pt[0] - start_pt[0]) * dy - (pt[1] - start_pt[1]) * dx) / line_len
            
            if dist > max_dist:
                max_dist = dist
                farthest_idx = j
        
        # Only include if depth is significant (> 1 pixel)
        if max_dist > 1:
            # fixpt_depth is depth * 256 (fixed point format)
            defects.append([start_idx, end_idx, farthest_idx, int(max_dist * 256)])
    
    if len(defects) == 0:
        return np.zeros((0, 1, 4), dtype=np.int32)
    
    return np.array(defects, dtype=np.int32).reshape(-1, 1, 4)


def fitLine(
    points: np.ndarray,
    distType: int,
    param: float,
    reps: float,
    aeps: float,
    line: Optional[np.ndarray] = None
) -> np.ndarray:
    """Fit a line to a 2D or 3D point set.
    
    Args:
        points: Input point set (Nx2 or Nx3)
        distType: Distance type (DIST_L2, DIST_L1, etc.)
        param: Numerical parameter for some distance types
        reps: Sufficient accuracy for radius
        aeps: Sufficient accuracy for angle
        line: Optional output line
    
    Returns:
        Output line [vx, vy, x0, y0] for 2D or [vx, vy, vz, x0, y0, z0] for 3D
    """
    points = np.asarray(points)
    if points.ndim == 3:
        points = points.reshape(-1, points.shape[-1])
    
    n_points = len(points)
    if n_points < 2:
        if points.shape[1] == 2:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    dim = points.shape[1]
    
    if distType == 2:  # DIST_L2 - use SVD
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD to find principal direction
        _, _, Vt = np.linalg.svd(centered)
        direction = Vt[0]
        direction = direction / np.linalg.norm(direction)
        
        if dim == 2:
            return np.array([direction[0], direction[1], centroid[0], centroid[1]], dtype=np.float32)
        else:
            return np.array([direction[0], direction[1], direction[2], 
                           centroid[0], centroid[1], centroid[2]], dtype=np.float32)
    
    else:
        # For other distance types, use iteratively reweighted least squares
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Initial direction from SVD
        _, _, Vt = np.linalg.svd(centered)
        direction = Vt[0]
        
        # Iterative refinement
        for _ in range(10):
            # Project points onto line
            projections = np.dot(centered, direction)
            residuals = centered - np.outer(projections, direction)
            distances = np.linalg.norm(residuals, axis=1)
            
            # Compute weights based on distance type
            if distType == 1:  # DIST_L1
                weights = 1.0 / (distances + 1e-6)
            elif distType == 4:  # DIST_L12
                weights = 1.0 / np.sqrt(1 + distances**2)
            elif distType == 5:  # DIST_FAIR
                c = param if param > 0 else 1.3998
                weights = 1.0 / (1 + distances / c)
            elif distType == 6:  # DIST_WELSCH
                c = param if param > 0 else 2.9846
                weights = np.exp(-(distances / c)**2)
            elif distType == 7:  # DIST_HUBER
                c = param if param > 0 else 1.345
                weights = np.where(distances <= c, 1.0, c / distances)
            else:
                weights = np.ones(n_points)
            
            # Weighted SVD
            weighted_centered = centered * weights[:, np.newaxis]
            _, _, Vt = np.linalg.svd(weighted_centered)
            direction = Vt[0]
            direction = direction / np.linalg.norm(direction)
        
        if dim == 2:
            return np.array([direction[0], direction[1], centroid[0], centroid[1]], dtype=np.float32)
        else:
            return np.array([direction[0], direction[1], direction[2],
                           centroid[0], centroid[1], centroid[2]], dtype=np.float32)


def fitEllipse(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Fit an ellipse around a set of 2D points.
    
    Args:
        points: Input 2D point set
    
    Returns:
        RotatedRect as ((center_x, center_y), (width, height), angle)
    """
    points = np.asarray(points)
    if points.ndim == 3:
        points = points.reshape(-1, 2)
    
    if len(points) < 5:
        # Not enough points for ellipse
        rect = boundingRect(points)
        cx = rect[0] + rect[2] / 2
        cy = rect[1] + rect[3] / 2
        return ((cx, cy), (float(rect[2]), float(rect[3])), 0.0)
    
    # Normalize points
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # Build design matrix for ellipse fitting
    x = centered[:, 0]
    y = centered[:, 1]
    
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])
    
    # Constraint matrix for ellipse
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = 2
    C[1, 1] = -1
    C[2, 0] = 2
    
    # Solve generalized eigenvalue problem
    try:
        from scipy.linalg import eig
        eigenvalues, eigenvectors = eig(S, C)
        # Find the positive eigenvalue
        idx = np.where(np.isfinite(eigenvalues) & (eigenvalues.real > 0))[0]
        if len(idx) == 0:
            idx = 0
        else:
            idx = idx[np.argmin(eigenvalues[idx].real)]
        params = eigenvectors[:, idx].real
    except:
        # Fallback to simple fitting
        _, _, Vt = np.linalg.svd(D)
        params = Vt[-1]
    
    # Extract ellipse parameters
    A, B, C_coef, D_coef, E_coef, F = params
    
    # Avoid division by zero
    if abs(A) < 1e-10 and abs(C_coef) < 1e-10:
        rect = boundingRect(points)
        cx = rect[0] + rect[2] / 2
        cy = rect[1] + rect[3] / 2
        return ((cx, cy), (float(rect[2]), float(rect[3])), 0.0)
    
    # Calculate ellipse center
    denom = 4 * A * C_coef - B**2
    if abs(denom) < 1e-10:
        denom = 1e-10
    
    cx = (B * E_coef - 2 * C_coef * D_coef) / denom
    cy = (B * D_coef - 2 * A * E_coef) / denom
    
    # Calculate angle
    if abs(A - C_coef) < 1e-10:
        angle = 45.0 if B > 0 else -45.0
    else:
        angle = 0.5 * np.arctan2(B, A - C_coef) * 180 / np.pi
    
    # Calculate axes
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))
    
    A_rot = A * cos_a**2 + B * cos_a * sin_a + C_coef * sin_a**2
    C_rot = A * sin_a**2 - B * cos_a * sin_a + C_coef * cos_a**2
    
    F_new = F + A * cx**2 + B * cx * cy + C_coef * cy**2 + D_coef * cx + E_coef * cy
    
    if abs(A_rot) > 1e-10:
        a = np.sqrt(abs(-F_new / A_rot))
    else:
        a = 1.0
    
    if abs(C_rot) > 1e-10:
        b = np.sqrt(abs(-F_new / C_rot))
    else:
        b = 1.0
    
    # Convert back to original coordinates
    cx += mean[0]
    cy += mean[1]
    
    return ((float(cx), float(cy)), (float(2*a), float(2*b)), float(angle))


def minEnclosingTriangle(points: np.ndarray) -> Tuple[float, np.ndarray]:
    """Find minimum area enclosing triangle.
    
    Args:
        points: Input 2D point set
    
    Returns:
        Tuple of (area, triangle_vertices)
    """
    points = np.asarray(points)
    if points.ndim == 3:
        points = points.reshape(-1, 2)
    
    if len(points) < 3:
        return 0.0, np.zeros((3, 1, 2), dtype=np.float32)
    
    # Get convex hull
    hull = convexHull(points, returnPoints=True)
    hull = hull.reshape(-1, 2)
    
    if len(hull) < 3:
        return 0.0, np.zeros((3, 1, 2), dtype=np.float32)
    
    # Simple approach: use rotating calipers (simplified)
    # For now, return bounding triangle based on minAreaRect
    rect = minAreaRect(points)
    
    center, (w, h), angle = rect
    angle_rad = np.radians(angle)
    
    # Create enclosing triangle from rotated rect
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Triangle vertices
    v1 = [center[0] - w * cos_a, center[1] - w * sin_a]
    v2 = [center[0] + w * cos_a, center[1] + w * sin_a]
    v3 = [center[0] - h * sin_a, center[1] + h * cos_a]
    
    triangle = np.array([[v1], [v2], [v3]], dtype=np.float32)
    
    # Calculate area
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))
    
    return float(area), triangle


# Distance type constants for fitLine
DIST_USER = -1
DIST_L1 = 1
DIST_L2 = 2
DIST_C = 3
DIST_L12 = 4
DIST_FAIR = 5
DIST_WELSCH = 6
DIST_HUBER = 7


# Exports

__all__ = [
    # Retrieval modes
    "RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE", "RETR_FLOODFILL",
    # Approximation methods
    "CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS",
    # Functions
    "findContours",
    "contourArea",
    "arcLength",
    "boundingRect",
    "minAreaRect",
    "minEnclosingCircle",
    "convexHull",
    "approxPolyDP",
    "moments",
    "isContourConvex",
    "pointPolygonTest",
    # New functions
    "convexityDefects",
    "fitLine",
    "fitEllipse",
    "minEnclosingTriangle",
    # Distance types
    "DIST_USER", "DIST_L1", "DIST_L2", "DIST_C", "DIST_L12", "DIST_FAIR", "DIST_WELSCH", "DIST_HUBER",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.