# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.features.matching - Feature Matching

Provides feature matching and visualization functions.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Union
import numpy as np

from .descriptors import KeyPoint


class DMatch:
    """Class for matching keypoint descriptors."""
    
    def __init__(
        self,
        queryIdx: int = -1,
        trainIdx: int = -1,
        imgIdx: int = -1,
        distance: float = float('inf')
    ):
        """Initialize match.
        
        Args:
            queryIdx: Query descriptor index
            trainIdx: Train descriptor index
            imgIdx: Train image index
            distance: Distance between descriptors
        """
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.imgIdx = imgIdx
        self.distance = distance
    
    def __repr__(self):
        return f"DMatch(queryIdx={self.queryIdx}, trainIdx={self.trainIdx}, distance={self.distance:.4f})"
    
    def __lt__(self, other):
        return self.distance < other.distance


class DescriptorMatcher:
    """Base class for descriptor matchers."""
    
    def match(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[DMatch]:
        """Find the best match for each descriptor."""
        raise NotImplementedError
    
    def knnMatch(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        k: int,
        mask: Optional[np.ndarray] = None,
        compactResult: bool = False
    ) -> List[List[DMatch]]:
        """Find k best matches for each descriptor."""
        raise NotImplementedError
    
    def radiusMatch(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        maxDistance: float,
        mask: Optional[np.ndarray] = None,
        compactResult: bool = False
    ) -> List[List[DMatch]]:
        """Find matches within a distance threshold."""
        raise NotImplementedError


class BFMatcher(DescriptorMatcher):
    """Brute-Force Descriptor Matcher."""
    
    def __init__(
        self,
        normType: int = 4,  # NORM_L2
        crossCheck: bool = False
    ):
        """Initialize BFMatcher.
        
        Args:
            normType: Norm type (NORM_L1=2, NORM_L2=4, NORM_HAMMING=6)
            crossCheck: If True, only return matches where both match each other
        """
        self._normType = normType
        self._crossCheck = crossCheck
    
    def _compute_distance(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> float:
        """Compute distance between two descriptors."""
        NORM_L1 = 2
        NORM_L2 = 4
        NORM_HAMMING = 6
        NORM_HAMMING2 = 7
        
        if self._normType == NORM_L1:
            return float(np.sum(np.abs(desc1.astype(np.float32) - desc2.astype(np.float32))))
        elif self._normType == NORM_L2:
            diff = desc1.astype(np.float32) - desc2.astype(np.float32)
            return float(np.sqrt(np.sum(diff * diff)))
        elif self._normType in (NORM_HAMMING, NORM_HAMMING2):
            # Hamming distance for binary descriptors
            xor = np.bitwise_xor(desc1, desc2)
            return float(np.unpackbits(xor).sum())
        else:
            # Default to L2
            diff = desc1.astype(np.float32) - desc2.astype(np.float32)
            return float(np.sqrt(np.sum(diff * diff)))
    
    def match(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[DMatch]:
        """Find the best match for each query descriptor.
        
        Args:
            queryDescriptors: Query descriptors (NxD)
            trainDescriptors: Train descriptors (MxD)
            mask: Optional mask
        
        Returns:
            List of best matches
        """
        if len(queryDescriptors) == 0 or len(trainDescriptors) == 0:
            return []
        
        matches = []
        
        for i, q_desc in enumerate(queryDescriptors):
            best_dist = float('inf')
            best_idx = -1
            
            for j, t_desc in enumerate(trainDescriptors):
                if mask is not None and not mask[i, j]:
                    continue
                
                dist = self._compute_distance(q_desc, t_desc)
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            
            if best_idx >= 0:
                matches.append(DMatch(i, best_idx, 0, best_dist))
        
        if self._crossCheck:
            # Filter by cross-check
            reverse_matches = {}
            for j, t_desc in enumerate(trainDescriptors):
                best_dist = float('inf')
                best_idx = -1
                
                for i, q_desc in enumerate(queryDescriptors):
                    dist = self._compute_distance(q_desc, t_desc)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                
                if best_idx >= 0:
                    reverse_matches[j] = best_idx
            
            matches = [m for m in matches 
                      if m.trainIdx in reverse_matches and 
                         reverse_matches[m.trainIdx] == m.queryIdx]
        
        return matches
    
    def knnMatch(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        k: int,
        mask: Optional[np.ndarray] = None,
        compactResult: bool = False
    ) -> List[List[DMatch]]:
        """Find k best matches for each query descriptor.
        
        Args:
            queryDescriptors: Query descriptors
            trainDescriptors: Train descriptors
            k: Number of matches to find
            mask: Optional mask
            compactResult: If True, exclude empty results
        
        Returns:
            List of k-best matches for each query
        """
        if len(queryDescriptors) == 0 or len(trainDescriptors) == 0:
            return []
        
        all_matches = []
        
        for i, q_desc in enumerate(queryDescriptors):
            distances = []
            
            for j, t_desc in enumerate(trainDescriptors):
                if mask is not None and not mask[i, j]:
                    continue
                
                dist = self._compute_distance(q_desc, t_desc)
                distances.append((j, dist))
            
            # Sort by distance and take top k
            distances.sort(key=lambda x: x[1])
            
            k_matches = []
            for idx, dist in distances[:k]:
                k_matches.append(DMatch(i, idx, 0, dist))
            
            if not compactResult or len(k_matches) > 0:
                all_matches.append(k_matches)
        
        return all_matches
    
    def radiusMatch(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        maxDistance: float,
        mask: Optional[np.ndarray] = None,
        compactResult: bool = False
    ) -> List[List[DMatch]]:
        """Find all matches within distance threshold.
        
        Args:
            queryDescriptors: Query descriptors
            trainDescriptors: Train descriptors
            maxDistance: Maximum distance threshold
            mask: Optional mask
            compactResult: If True, exclude empty results
        
        Returns:
            List of matches within threshold for each query
        """
        if len(queryDescriptors) == 0 or len(trainDescriptors) == 0:
            return []
        
        all_matches = []
        
        for i, q_desc in enumerate(queryDescriptors):
            radius_matches = []
            
            for j, t_desc in enumerate(trainDescriptors):
                if mask is not None and not mask[i, j]:
                    continue
                
                dist = self._compute_distance(q_desc, t_desc)
                
                if dist <= maxDistance:
                    radius_matches.append(DMatch(i, j, 0, dist))
            
            radius_matches.sort(key=lambda m: m.distance)
            
            if not compactResult or len(radius_matches) > 0:
                all_matches.append(radius_matches)
        
        return all_matches


class FlannBasedMatcher(DescriptorMatcher):
    """FLANN-based Descriptor Matcher.
    
    Note: Uses approximate nearest neighbor search for speed.
    """
    
    def __init__(
        self,
        indexParams: Optional[dict] = None,
        searchParams: Optional[dict] = None
    ):
        """Initialize FlannBasedMatcher.
        
        Args:
            indexParams: Index parameters
            searchParams: Search parameters
        """
        self._indexParams = indexParams or {'algorithm': 1, 'trees': 5}
        self._searchParams = searchParams or {'checks': 50}
        
        # Internal BFMatcher as fallback
        self._bf = BFMatcher(normType=4, crossCheck=False)
    
    def match(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[DMatch]:
        """Find best match for each descriptor."""
        # Use BFMatcher as simplified implementation
        return self._bf.match(queryDescriptors, trainDescriptors, mask)
    
    def knnMatch(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        k: int,
        mask: Optional[np.ndarray] = None,
        compactResult: bool = False
    ) -> List[List[DMatch]]:
        """Find k best matches."""
        return self._bf.knnMatch(queryDescriptors, trainDescriptors, k, mask, compactResult)
    
    def radiusMatch(
        self,
        queryDescriptors: np.ndarray,
        trainDescriptors: np.ndarray,
        maxDistance: float,
        mask: Optional[np.ndarray] = None,
        compactResult: bool = False
    ) -> List[List[DMatch]]:
        """Find matches within radius."""
        return self._bf.radiusMatch(queryDescriptors, trainDescriptors, maxDistance, mask, compactResult)


def drawKeypoints(
    image: np.ndarray,
    keypoints: List[KeyPoint],
    outImage: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    flags: int = 0
) -> np.ndarray:
    """Draw keypoints on an image.
    
    Args:
        image: Input image
        keypoints: Keypoints to draw
        outImage: Optional output image
        color: Keypoint color (B, G, R)
        flags: Drawing flags
    
    Returns:
        Image with keypoints drawn
    """
    DRAW_MATCHES_FLAGS_DEFAULT = 0
    DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG = 1
    DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS = 2
    DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS = 4
    
    if outImage is None or not (flags & DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG):
        if image.ndim == 2:
            result = np.stack([image, image, image], axis=2)
        else:
            result = image.copy()
    else:
        result = outImage
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Random color if color is (-1, -1, -1)
        if color == (-1, -1, -1):
            c = tuple(np.random.randint(0, 255, 3).tolist())
        else:
            c = color
        
        if flags & DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS:
            # Draw circle with size and orientation
            radius = int(kp.size / 2)
            
            # Draw circle
            for angle in range(360):
                px = int(x + radius * np.cos(np.radians(angle)))
                py = int(y + radius * np.sin(np.radians(angle)))
                if 0 <= px < result.shape[1] and 0 <= py < result.shape[0]:
                    result[py, px] = c
            
            # Draw orientation line
            if kp.angle >= 0:
                ex = int(x + radius * np.cos(np.radians(kp.angle)))
                ey = int(y + radius * np.sin(np.radians(kp.angle)))
                
                # Simple line drawing
                steps = max(abs(ex - x), abs(ey - y)) + 1
                for t in np.linspace(0, 1, steps):
                    lx = int(x + t * (ex - x))
                    ly = int(y + t * (ey - y))
                    if 0 <= lx < result.shape[1] and 0 <= ly < result.shape[0]:
                        result[ly, lx] = c
        else:
            # Simple point
            radius = 3
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        px, py = x + dx, y + dy
                        if 0 <= px < result.shape[1] and 0 <= py < result.shape[0]:
                            result[py, px] = c
    
    return result


def drawMatches(
    img1: np.ndarray,
    keypoints1: List[KeyPoint],
    img2: np.ndarray,
    keypoints2: List[KeyPoint],
    matches: List[DMatch],
    outImg: Optional[np.ndarray] = None,
    matchColor: Tuple[int, int, int] = (0, 255, 0),
    singlePointColor: Tuple[int, int, int] = (255, 0, 0),
    matchesMask: Optional[List[int]] = None,
    flags: int = 0
) -> np.ndarray:
    """Draw matches between two images.
    
    Args:
        img1: First image
        keypoints1: Keypoints from first image
        img2: Second image
        keypoints2: Keypoints from second image
        matches: Matches to draw
        outImg: Optional output image
        matchColor: Color for matches
        singlePointColor: Color for unmatched keypoints
        matchesMask: Mask specifying which matches to draw
        flags: Drawing flags
    
    Returns:
        Image with matches drawn
    """
    # Ensure both images are color
    if img1.ndim == 2:
        img1_color = np.stack([img1, img1, img1], axis=2)
    else:
        img1_color = img1.copy()
    
    if img2.ndim == 2:
        img2_color = np.stack([img2, img2, img2], axis=2)
    else:
        img2_color = img2.copy()
    
    # Create output image (side by side)
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    
    max_h = max(h1, h2)
    result = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    result[:h1, :w1] = img1_color
    result[:h2, w1:w1+w2] = img2_color
    
    # Draw matches
    for i, m in enumerate(matches):
        if matchesMask is not None and not matchesMask[i]:
            continue
        
        kp1 = keypoints1[m.queryIdx]
        kp2 = keypoints2[m.trainIdx]
        
        pt1 = (int(kp1.pt[0]), int(kp1.pt[1]))
        pt2 = (int(kp2.pt[0]) + w1, int(kp2.pt[1]))
        
        # Random color if (-1, -1, -1)
        if matchColor == (-1, -1, -1):
            color = tuple(np.random.randint(0, 255, 3).tolist())
        else:
            color = matchColor
        
        # Draw line
        steps = max(abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1])) + 1
        for t in np.linspace(0, 1, steps):
            x = int(pt1[0] + t * (pt2[0] - pt1[0]))
            y = int(pt1[1] + t * (pt2[1] - pt1[1]))
            if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                result[y, x] = color
        
        # Draw keypoint circles
        for pt in [pt1, pt2]:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx*dx + dy*dy <= 9:
                        px, py = pt[0] + dx, pt[1] + dy
                        if 0 <= px < result.shape[1] and 0 <= py < result.shape[0]:
                            result[py, px] = color
    
    return result


def drawMatchesKnn(
    img1: np.ndarray,
    keypoints1: List[KeyPoint],
    img2: np.ndarray,
    keypoints2: List[KeyPoint],
    matches: List[List[DMatch]],
    outImg: Optional[np.ndarray] = None,
    matchColor: Tuple[int, int, int] = (0, 255, 0),
    singlePointColor: Tuple[int, int, int] = (255, 0, 0),
    matchesMask: Optional[List[List[int]]] = None,
    flags: int = 0
) -> np.ndarray:
    """Draw k-nearest matches.
    
    Args:
        img1: First image
        keypoints1: Keypoints from first image
        img2: Second image
        keypoints2: Keypoints from second image
        matches: K-nearest matches
        outImg: Optional output image
        matchColor: Color for matches
        singlePointColor: Color for keypoints
        matchesMask: Mask for which matches to draw
        flags: Drawing flags
    
    Returns:
        Image with matches drawn
    """
    # Flatten matches and mask
    flat_matches = []
    flat_mask = None
    
    if matchesMask is not None:
        flat_mask = []
    
    for i, knn_match in enumerate(matches):
        for j, m in enumerate(knn_match):
            flat_matches.append(m)
            if matchesMask is not None:
                if i < len(matchesMask) and j < len(matchesMask[i]):
                    flat_mask.append(matchesMask[i][j])
                else:
                    flat_mask.append(0)
    
    return drawMatches(img1, keypoints1, img2, keypoints2, flat_matches,
                       outImg, matchColor, singlePointColor, flat_mask, flags)


# Norm types
NORM_INF = 1
NORM_L1 = 2
NORM_L2 = 4
NORM_L2SQR = 5
NORM_HAMMING = 6
NORM_HAMMING2 = 7

# Draw flags
DRAW_MATCHES_FLAGS_DEFAULT = 0
DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG = 1
DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS = 2
DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS = 4


def BFMatcher_create(
    normType: int = NORM_L2,
    crossCheck: bool = False
) -> BFMatcher:
    """Create BFMatcher."""
    return BFMatcher(normType, crossCheck)


def FlannBasedMatcher_create(
    indexParams: Optional[dict] = None,
    searchParams: Optional[dict] = None
) -> FlannBasedMatcher:
    """Create FlannBasedMatcher."""
    return FlannBasedMatcher(indexParams, searchParams)


__all__ = [
    "DMatch",
    "DescriptorMatcher",
    "BFMatcher",
    "FlannBasedMatcher",
    "BFMatcher_create",
    "FlannBasedMatcher_create",
    "drawKeypoints",
    "drawMatches",
    "drawMatchesKnn",
    
    # Constants
    "NORM_INF",
    "NORM_L1",
    "NORM_L2",
    "NORM_L2SQR",
    "NORM_HAMMING",
    "NORM_HAMMING2",
    "DRAW_MATCHES_FLAGS_DEFAULT",
    "DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG",
    "DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS",
    "DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.