# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.imgproc.drawing - Drawing functions for images

Provides Neurova drawing functions for shapes, lines, and text.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

# Line types
LINE_4 = 4
LINE_8 = 8
LINE_AA = 16  # Anti-aliased

# Marker types
MARKER_CROSS = 0
MARKER_TILTED_CROSS = 1
MARKER_STAR = 2
MARKER_DIAMOND = 3
MARKER_SQUARE = 4
MARKER_TRIANGLE_UP = 5
MARKER_TRIANGLE_DOWN = 6

# Font faces
FONT_HERSHEY_SIMPLEX = 0
FONT_HERSHEY_PLAIN = 1
FONT_HERSHEY_DUPLEX = 2
FONT_HERSHEY_COMPLEX = 3
FONT_HERSHEY_TRIPLEX = 4
FONT_HERSHEY_COMPLEX_SMALL = 5
FONT_HERSHEY_SCRIPT_SIMPLEX = 6
FONT_HERSHEY_SCRIPT_COMPLEX = 7
FONT_ITALIC = 16

# Fill modes
FILLED = -1

# Type aliases
Point = Union[Tuple[int, int], Tuple[float, float], List[int], List[float]]
Color = Union[Tuple[int, ...], List[int], int, float]


def _to_int_point(pt: Point) -> Tuple[int, int]:
    """Convert point to integer coordinates."""
    return (int(pt[0]), int(pt[1]))


def _to_color(color: Color, channels: int) -> Tuple[int, ...]:
    """Convert color to tuple of integers."""
    if isinstance(color, (int, float)):
        return tuple([int(color)] * channels)
    return tuple(int(c) for c in color[:channels])


def _clip_line(pt1: Tuple[int, int], pt2: Tuple[int, int], 
               width: int, height: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Clip line to image bounds using Cohen-Sutherland algorithm."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Simple bounds check for now
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    
    return ((x1, y1), (x2, y2))


def line(
    img: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    shift: int = 0
) -> np.ndarray:
    """Draw a line segment connecting two points.
    
    Args:
        img: Image to draw on
        pt1: First point (x, y)
        pt2: Second point (x, y)
        color: Line color
        thickness: Line thickness
        lineType: Type of line (LINE_4, LINE_8, LINE_AA)
        shift: Fractional bit shift
    
    Returns:
        Image with drawn line
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    
    x1, y1 = _to_int_point(pt1)
    x2, y2 = _to_int_point(pt2)
    col = _to_color(color, channels)
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    thickness = max(1, thickness)
    half_t = thickness // 2
    
    x, y = x1, y1
    while True:
        # Draw with thickness
        for tx in range(-half_t, half_t + 1):
            for ty in range(-half_t, half_t + 1):
                px, py = x + tx, y + ty
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = col
        
        if x == x2 and y == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return img


def arrowedLine(
    img: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    shift: int = 0,
    tipLength: float = 0.1
) -> np.ndarray:
    """Draw an arrow segment pointing from pt1 to pt2.
    
    Args:
        img: Image to draw on
        pt1: Start point
        pt2: End point (arrow tip)
        color: Arrow color
        thickness: Line thickness
        lineType: Type of line
        shift: Fractional bit shift
        tipLength: Length of arrow tip relative to arrow length
    
    Returns:
        Image with drawn arrow
    """
    x1, y1 = _to_int_point(pt1)
    x2, y2 = _to_int_point(pt2)
    
    # Draw main line
    line(img, pt1, pt2, color, thickness, lineType, shift)
    
    # Calculate arrow tips
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if length == 0:
        return img
    
    tip_len = length * tipLength
    angle = np.arctan2(y2 - y1, x2 - x1)
    
    # Arrow tip angles (30 degrees from main line)
    angle1 = angle + np.pi * 5 / 6
    angle2 = angle - np.pi * 5 / 6
    
    tip1 = (int(x2 + tip_len * np.cos(angle1)),
            int(y2 + tip_len * np.sin(angle1)))
    tip2 = (int(x2 + tip_len * np.cos(angle2)),
            int(y2 + tip_len * np.sin(angle2)))
    
    line(img, pt2, tip1, color, thickness, lineType, shift)
    line(img, pt2, tip2, color, thickness, lineType, shift)
    
    return img


def rectangle(
    img: np.ndarray,
    pt1: Point,
    pt2: Point,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    shift: int = 0
) -> np.ndarray:
    """Draw a rectangle.
    
    Args:
        img: Image to draw on
        pt1: Top-left corner (or one corner)
        pt2: Bottom-right corner (or opposite corner)
        color: Rectangle color
        thickness: Line thickness (-1 for filled)
        lineType: Type of line
        shift: Fractional bit shift
    
    Returns:
        Image with drawn rectangle
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    
    x1, y1 = _to_int_point(pt1)
    x2, y2 = _to_int_point(pt2)
    col = _to_color(color, channels)
    
    # Ensure proper ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Clip to image bounds
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    
    if thickness < 0 or thickness == FILLED:
        # Filled rectangle
        img[y1:y2+1, x1:x2+1] = col
    else:
        # Draw outline
        t = max(1, thickness)
        # Top and bottom edges
        for i in range(t):
            if y1 + i <= y2:
                img[y1 + i, x1:x2+1] = col
            if y2 - i >= y1:
                img[y2 - i, x1:x2+1] = col
        # Left and right edges
        for i in range(t):
            if x1 + i <= x2:
                img[y1:y2+1, x1 + i] = col
            if x2 - i >= x1:
                img[y1:y2+1, x2 - i] = col
    
    return img


def circle(
    img: np.ndarray,
    center: Point,
    radius: int,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    shift: int = 0
) -> np.ndarray:
    """Draw a circle.
    
    Args:
        img: Image to draw on
        center: Center of the circle
        radius: Radius of the circle
        color: Circle color
        thickness: Line thickness (-1 for filled)
        lineType: Type of line
        shift: Fractional bit shift
    
    Returns:
        Image with drawn circle
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    
    cx, cy = _to_int_point(center)
    col = _to_color(color, channels)
    
    if thickness < 0 or thickness == FILLED:
        # Filled circle using distance formula
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist <= radius
        img[mask] = col
    else:
        # Draw circle outline using midpoint algorithm
        t = max(1, thickness)
        
        for r in range(max(0, radius - t // 2), radius + t // 2 + 1):
            x = 0
            y = r
            d = 3 - 2 * r
            
            while y >= x:
                # Draw 8 octants
                for dx, dy in [(x, y), (y, x), (-x, y), (-y, x),
                               (x, -y), (y, -x), (-x, -y), (-y, -x)]:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = col
                
                x += 1
                if d > 0:
                    y -= 1
                    d = d + 4 * (x - y) + 10
                else:
                    d = d + 4 * x + 6
    
    return img


def ellipse(
    img: np.ndarray,
    center: Point,
    axes: Tuple[int, int],
    angle: float,
    startAngle: float,
    endAngle: float,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    shift: int = 0
) -> np.ndarray:
    """Draw an ellipse or elliptic arc.
    
    Args:
        img: Image to draw on
        center: Center of the ellipse
        axes: Half axes (width/2, height/2)
        angle: Rotation angle in degrees
        startAngle: Starting angle in degrees
        endAngle: Ending angle in degrees
        color: Ellipse color
        thickness: Line thickness (-1 for filled)
        lineType: Type of line
        shift: Fractional bit shift
    
    Returns:
        Image with drawn ellipse
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    
    cx, cy = _to_int_point(center)
    ax, ay = int(axes[0]), int(axes[1])
    col = _to_color(color, channels)
    
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    start_rad = np.radians(startAngle)
    end_rad = np.radians(endAngle)
    
    # Number of points to draw
    num_points = max(100, int(max(ax, ay) * 2 * np.pi / 4))
    
    if thickness < 0 or thickness == FILLED:
        # Filled ellipse
        y, x = np.ogrid[:h, :w]
        # Transform coordinates
        dx = x - cx
        dy = y - cy
        # Rotate back
        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a
        # Check if inside ellipse
        if ax > 0 and ay > 0:
            mask = (rx ** 2 / ax ** 2 + ry ** 2 / ay ** 2) <= 1
            img[mask] = col
    else:
        # Draw ellipse arc
        t = max(1, thickness)
        for angle_t in np.linspace(start_rad, end_rad, num_points):
            # Point on ellipse before rotation
            ex = ax * np.cos(angle_t)
            ey = ay * np.sin(angle_t)
            # Rotate
            rx = cx + ex * cos_a - ey * sin_a
            ry = cy + ex * sin_a + ey * cos_a
            
            # Draw with thickness
            for tx in range(-t // 2, t // 2 + 1):
                for ty in range(-t // 2, t // 2 + 1):
                    px, py = int(rx) + tx, int(ry) + ty
                    if 0 <= px < w and 0 <= py < h:
                        img[py, px] = col
    
    return img


def polylines(
    img: np.ndarray,
    pts: Sequence[np.ndarray],
    isClosed: bool,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    shift: int = 0
) -> np.ndarray:
    """Draw polygonal curves.
    
    Args:
        img: Image to draw on
        pts: Array of polygonal curves (list of point arrays)
        isClosed: Whether to close the curves
        color: Line color
        thickness: Line thickness
        lineType: Type of line
        shift: Fractional bit shift
    
    Returns:
        Image with drawn polylines
    """
    for poly in pts:
        points = np.asarray(poly).reshape(-1, 2)
        n = len(points)
        
        for i in range(n - 1):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + 1])
            line(img, pt1, pt2, color, thickness, lineType, shift)
        
        if isClosed and n > 1:
            line(img, tuple(points[-1]), tuple(points[0]), 
                 color, thickness, lineType, shift)
    
    return img


def fillPoly(
    img: np.ndarray,
    pts: Sequence[np.ndarray],
    color: Color,
    lineType: int = LINE_8,
    shift: int = 0,
    offset: Point = (0, 0)
) -> np.ndarray:
    """Fill polygons.
    
    Args:
        img: Image to draw on
        pts: Array of polygons (list of point arrays)
        color: Fill color
        lineType: Type of line
        shift: Fractional bit shift
        offset: Offset for all points
    
    Returns:
        Image with filled polygons
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    col = _to_color(color, channels)
    ox, oy = _to_int_point(offset)
    
    for poly in pts:
        points = np.asarray(poly).reshape(-1, 2).astype(int)
        points[:, 0] += ox
        points[:, 1] += oy
        
        # Scanline fill algorithm
        if len(points) < 3:
            continue
        
        min_y = max(0, points[:, 1].min())
        max_y = min(h - 1, points[:, 1].max())
        
        for y in range(min_y, max_y + 1):
            # Find intersections with polygon edges
            intersections = []
            n = len(points)
            
            for i in range(n):
                p1 = points[i]
                p2 = points[(i + 1) % n]
                
                if p1[1] == p2[1]:
                    continue
                
                if min(p1[1], p2[1]) <= y < max(p1[1], p2[1]):
                    x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                    intersections.append(int(x))
            
            intersections.sort()
            
            # Fill between pairs of intersections
            for i in range(0, len(intersections) - 1, 2):
                x1 = max(0, intersections[i])
                x2 = min(w - 1, intersections[i + 1])
                if x1 <= x2:
                    img[y, x1:x2+1] = col
    
    return img


def fillConvexPoly(
    img: np.ndarray,
    points: np.ndarray,
    color: Color,
    lineType: int = LINE_8,
    shift: int = 0
) -> np.ndarray:
    """Fill a convex polygon.
    
    Args:
        img: Image to draw on
        points: Polygon vertices
        color: Fill color
        lineType: Type of line
        shift: Fractional bit shift
    
    Returns:
        Image with filled polygon
    """
    return fillPoly(img, [points], color, lineType, shift)


def putText(
    img: np.ndarray,
    text: str,
    org: Point,
    fontFace: int,
    fontScale: float,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    bottomLeftOrigin: bool = False
) -> np.ndarray:
    """Draw a text string.
    
    Args:
        img: Image to draw on
        text: Text string to draw
        org: Bottom-left corner of the text string
        fontFace: Font type
        fontScale: Font scale factor
        color: Text color
        thickness: Line thickness
        lineType: Type of line
        bottomLeftOrigin: If true, origin is at bottom-left
    
    Returns:
        Image with drawn text
    """
    channels = img.shape[2] if img.ndim == 3 else 1
    col = _to_color(color, channels)
    x, y = _to_int_point(org)
    
    # Simple bitmap font rendering
    char_width = int(8 * fontScale)
    char_height = int(12 * fontScale)
    
    if bottomLeftOrigin:
        y = img.shape[0] - y
    
    for i, char in enumerate(text):
        cx = x + i * char_width
        
        # Draw a simple rectangle for each character (placeholder)
        # In a real implementation, you'd use a proper font
        if char != ' ':
            for dy in range(char_height):
                for dx in range(char_width - 1):
                    px, py = cx + dx, y - char_height + dy
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        # Simple character pattern based on ASCII
                        if _should_draw_pixel(char, dx, dy, char_width, char_height):
                            for t in range(max(1, thickness)):
                                if 0 <= py + t < img.shape[0]:
                                    img[py + t, px] = col
    
    return img


def _should_draw_pixel(char: str, dx: int, dy: int, w: int, h: int) -> bool:
    """Determine if a pixel should be drawn for a character."""
    # Simple 5x7 font patterns for common characters
    nw, nh = max(1, w // 2), max(1, h // 3)
    rx, ry = dx / max(1, w), dy / max(1, h)
    
    if char.isalpha():
        # Draw letter outline
        if ry < 0.1 or ry > 0.9:  # Top/bottom
            return 0.2 < rx < 0.8
        if rx < 0.15 or rx > 0.85:  # Sides
            return True
        if 0.45 < ry < 0.55:  # Middle
            return char.upper() in "ABEFHPRS"
    elif char.isdigit():
        # Simple digit patterns
        return (ry < 0.15 or ry > 0.85 or 0.45 < ry < 0.55 or 
                rx < 0.2 or rx > 0.8)
    elif char in ".:,;":
        return 0.4 < rx < 0.6 and (0.7 < ry < 0.9 or (char == ':' and 0.3 < ry < 0.5))
    elif char == '-':
        return 0.45 < ry < 0.55 and 0.2 < rx < 0.8
    elif char == '_':
        return ry > 0.9 and 0.1 < rx < 0.9
    elif char in "()[]{}":
        return rx < 0.3 or rx > 0.7 or ry < 0.1 or ry > 0.9
    
    return False


def getTextSize(
    text: str,
    fontFace: int,
    fontScale: float,
    thickness: int
) -> Tuple[Tuple[int, int], int]:
    """Calculate size of a text string.
    
    Args:
        text: Text string
        fontFace: Font type
        fontScale: Font scale factor
        thickness: Line thickness
    
    Returns:
        Tuple ((width, height), baseline)
    """
    char_width = int(8 * fontScale)
    char_height = int(12 * fontScale)
    baseline = int(3 * fontScale)
    
    width = len(text) * char_width
    height = char_height
    
    return ((width, height), baseline)


def drawMarker(
    img: np.ndarray,
    position: Point,
    color: Color,
    markerType: int = MARKER_CROSS,
    markerSize: int = 20,
    thickness: int = 1,
    line_type: int = LINE_8
) -> np.ndarray:
    """Draw a marker on an image.
    
    Args:
        img: Image to draw on
        position: Position of the marker
        color: Marker color
        markerType: Type of marker
        markerSize: Size of the marker
        thickness: Line thickness
        line_type: Type of line
    
    Returns:
        Image with drawn marker
    """
    x, y = _to_int_point(position)
    half = markerSize // 2
    
    if markerType == MARKER_CROSS:
        line(img, (x - half, y), (x + half, y), color, thickness, line_type)
        line(img, (x, y - half), (x, y + half), color, thickness, line_type)
    elif markerType == MARKER_TILTED_CROSS:
        d = int(half * 0.707)
        line(img, (x - d, y - d), (x + d, y + d), color, thickness, line_type)
        line(img, (x - d, y + d), (x + d, y - d), color, thickness, line_type)
    elif markerType == MARKER_STAR:
        line(img, (x - half, y), (x + half, y), color, thickness, line_type)
        line(img, (x, y - half), (x, y + half), color, thickness, line_type)
        d = int(half * 0.707)
        line(img, (x - d, y - d), (x + d, y + d), color, thickness, line_type)
        line(img, (x - d, y + d), (x + d, y - d), color, thickness, line_type)
    elif markerType == MARKER_DIAMOND:
        pts = np.array([[x, y - half], [x + half, y], 
                        [x, y + half], [x - half, y]])
        polylines(img, [pts], True, color, thickness, line_type)
    elif markerType == MARKER_SQUARE:
        rectangle(img, (x - half, y - half), (x + half, y + half), 
                  color, thickness, line_type)
    elif markerType == MARKER_TRIANGLE_UP:
        pts = np.array([[x, y - half], [x + half, y + half], [x - half, y + half]])
        polylines(img, [pts], True, color, thickness, line_type)
    elif markerType == MARKER_TRIANGLE_DOWN:
        pts = np.array([[x, y + half], [x + half, y - half], [x - half, y - half]])
        polylines(img, [pts], True, color, thickness, line_type)
    
    return img


def drawContours(
    image: np.ndarray,
    contours: Sequence[np.ndarray],
    contourIdx: int,
    color: Color,
    thickness: int = 1,
    lineType: int = LINE_8,
    hierarchy: Optional[np.ndarray] = None,
    maxLevel: int = 2147483647,
    offset: Point = (0, 0)
) -> np.ndarray:
    """Draw contours on an image.
    
    Args:
        image: Destination image
        contours: List of contours
        contourIdx: Index of contour to draw (-1 for all)
        color: Contour color
        thickness: Line thickness (-1 for filled)
        lineType: Type of line
        hierarchy: Contour hierarchy
        maxLevel: Maximum hierarchy level
        offset: Offset for all contour points
    
    Returns:
        Image with drawn contours
    """
    if contourIdx == -1:
        indices = range(len(contours))
    else:
        indices = [contourIdx]
    
    ox, oy = _to_int_point(offset)
    
    for idx in indices:
        if idx < 0 or idx >= len(contours):
            continue
        
        contour = contours[idx]
        points = np.asarray(contour).reshape(-1, 2)
        points = points + np.array([ox, oy])
        
        if thickness < 0 or thickness == FILLED:
            fillPoly(image, [points], color, lineType)
        else:
            polylines(image, [points], True, color, thickness, lineType)
    
    return image


# Exports

__all__ = [
    # Constants
    "LINE_4",
    "LINE_8",
    "LINE_AA",
    "MARKER_CROSS",
    "MARKER_TILTED_CROSS",
    "MARKER_STAR",
    "MARKER_DIAMOND",
    "MARKER_SQUARE",
    "MARKER_TRIANGLE_UP",
    "MARKER_TRIANGLE_DOWN",
    "FONT_HERSHEY_SIMPLEX",
    "FONT_HERSHEY_PLAIN",
    "FONT_HERSHEY_DUPLEX",
    "FONT_HERSHEY_COMPLEX",
    "FONT_HERSHEY_TRIPLEX",
    "FONT_HERSHEY_COMPLEX_SMALL",
    "FONT_HERSHEY_SCRIPT_SIMPLEX",
    "FONT_HERSHEY_SCRIPT_COMPLEX",
    "FONT_ITALIC",
    "FILLED",
    # Functions
    "line",
    "arrowedLine",
    "rectangle",
    "circle",
    "ellipse",
    "polylines",
    "fillPoly",
    "fillConvexPoly",
    "putText",
    "getTextSize",
    "drawMarker",
    "drawContours",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.