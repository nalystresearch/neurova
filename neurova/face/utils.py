# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Face processing utilities.

Helper functions for face detection, alignment, cropping, and visualization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np


def extract_faces(
    image: np.ndarray,
    detector: Optional[Any] = None,
    method: str = "haar",
    margin: float = 0.2,
    min_size: Tuple[int, int] = (30, 30),
    output_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Extract all faces from an image.
    
    Args:
        image: Input image (BGR or RGB numpy array).
        detector: FaceDetector instance (creates one if None).
        method: Detection method if creating detector.
        margin: Margin around face (fraction of face size).
        min_size: Minimum face size to detect.
        output_size: Resize output faces to this size.
    
    Returns:
        List of cropped face images.
        
    Example:
        >>> faces = extract_faces(image, method='haar')
        >>> for i, face in enumerate(faces):
        ...     save_image(face, f'face_{i}.jpg')
    """
    from .detector import FaceDetector
    
    if detector is None:
        detector = FaceDetector(method=method, min_size=min_size)
    
    return detector.detect_and_crop(image, margin=margin, size=output_size)


def align_face(
    face: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    desired_left_eye: Tuple[float, float] = (0.35, 0.35),
    desired_face_width: int = 256,
    desired_face_height: Optional[int] = None,
) -> np.ndarray:
    """
    Align a face image based on eye positions.
    
    Args:
        face: Input face image.
        landmarks: Face landmarks (68 points) or None to detect.
        desired_left_eye: Desired position of left eye (normalized).
        desired_face_width: Output face width.
        desired_face_height: Output face height (defaults to width).
    
    Returns:
        Aligned face image.
    """
    from PIL import Image
    
    if desired_face_height is None:
        desired_face_height = desired_face_width
    
    if landmarks is None:
        # Try to detect landmarks
        landmarks = detect_landmarks(face)
        if landmarks is None:
            # No landmarks, just resize
            pil_img = Image.fromarray(face)
            pil_img = pil_img.resize((desired_face_width, desired_face_height))
            return np.array(pil_img)
    
    # Get eye centers (assuming 68-point landmarks)
    if len(landmarks) >= 68:
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
    elif len(landmarks) >= 10:
        left_eye = landmarks[0]
        right_eye = landmarks[1]
    else:
        pil_img = Image.fromarray(face)
        pil_img = pil_img.resize((desired_face_width, desired_face_height))
        return np.array(pil_img)
    
    # Calculate angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Calculate scale
    dist = np.sqrt(dx**2 + dy**2)
    desired_dist = (1.0 - 2 * desired_left_eye[0]) * desired_face_width
    scale = desired_dist / dist
    
    # Calculate center
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)
    
    # Rotation matrix
    cos = np.cos(np.radians(angle))
    sin = np.sin(np.radians(angle))
    
    M = np.array([
        [cos * scale, -sin * scale, 
         eyes_center[0] * (1 - cos * scale) + eyes_center[1] * sin * scale],
        [sin * scale, cos * scale,
         eyes_center[1] * (1 - cos * scale) - eyes_center[0] * sin * scale]
    ])
    
    # Adjust for desired eye position
    M[0, 2] += (desired_face_width * 0.5 - eyes_center[0])
    M[1, 2] += (desired_face_height * desired_left_eye[1] - eyes_center[1])
    
    # Apply transformation
    try:
        import cv2
        output = cv2.warpAffine(face, M, (desired_face_width, desired_face_height))
    except ImportError:
        # Fallback to PIL
        pil_img = Image.fromarray(face)
        pil_img = pil_img.rotate(-angle, center=tuple(eyes_center), resample=Image.BILINEAR)
        pil_img = pil_img.resize((desired_face_width, desired_face_height))
        output = np.array(pil_img)
    
    return output


def detect_landmarks(
    face: np.ndarray,
    model_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Detect facial landmarks in a face image.
    
    Args:
        face: Face image.
        model_path: Path to landmark model.
    
    Returns:
        Array of (x, y) landmark coordinates or None.
    """
    # Try MediaPipe
    try:
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(face)
            
            if results.multi_face_landmarks:
                h, w = face.shape[:2]
                landmarks = []
                for lm in results.multi_face_landmarks[0].landmark:
                    landmarks.append([lm.x * w, lm.y * h])
                return np.array(landmarks)
    except ImportError:
        pass
    
    # Try dlib
    try:
        import dlib
        
        if model_path is None:
            # Use bundled model
            data_dir = Path(__file__).resolve().parent.parent / "data"
            model_path = data_dir / "shape_predictor_68_face_landmarks.dat"
            
            if not model_path.exists():
                return None
        
        predictor = dlib.shape_predictor(str(model_path))
        detector = dlib.get_frontal_face_detector()
        
        if len(face.shape) == 3:
            gray = np.mean(face, axis=2).astype(np.uint8)
        else:
            gray = face
        
        faces = detector(gray)
        if len(faces) > 0:
            shape = predictor(gray, faces[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            return landmarks
    except ImportError:
        pass
    
    return None


def crop_face(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    margin: float = 0.0,
    output_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Crop a face from an image given bounding box.
    
    Args:
        image: Input image.
        box: Bounding box (x, y, width, height).
        margin: Margin around face (fraction of face size).
        output_size: Resize to this size.
    
    Returns:
        Cropped face image.
    """
    from PIL import Image
    
    x, y, w, h = box
    ih, iw = image.shape[:2]
    
    # Add margin
    mx = int(w * margin)
    my = int(h * margin)
    
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(iw, x + w + mx)
    y2 = min(ih, y + h + my)
    
    cropped = image[y1:y2, x1:x2].copy()
    
    if output_size and cropped.size > 0:
        pil_img = Image.fromarray(cropped)
        pil_img = pil_img.resize(output_size, Image.LANCZOS)
        cropped = np.array(pil_img)
    
    return cropped


def draw_faces(
    image: np.ndarray,
    faces: List[Tuple[int, int, int, int, float]],
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True,
    copy: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes around detected faces.
    
    Args:
        image: Input image.
        faces: List of (x, y, w, h, confidence) tuples.
        labels: Optional labels for each face.
        color: Box color (BGR).
        thickness: Line thickness.
        font_scale: Font size for labels.
        show_confidence: Show confidence scores.
        copy: Whether to copy the image first.
    
    Returns:
        Image with drawn faces.
    """
    if copy:
        result = image.copy()
    else:
        result = image
    
    for i, (x, y, w, h, conf) in enumerate(faces):
        # Draw rectangle
        _draw_rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Prepare label
        label = ""
        if labels and i < len(labels):
            label = labels[i]
        if show_confidence:
            if label:
                label = f"{label} ({conf:.2f})"
            else:
                label = f"{conf:.2f}"
        
        if label:
            _draw_text(result, label, (x, y - 10), color, font_scale)
    
    return result


def _draw_rectangle(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw rectangle on image."""
    try:
        import cv2
        cv2.rectangle(image, pt1, pt2, color, thickness)
    except ImportError:
        # Manual rectangle drawing
        x1, y1 = pt1
        x2, y2 = pt2
        h, w = image.shape[:2]
        
        # Clamp coordinates
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        
        # Draw lines
        for t in range(thickness):
            # Top
            if y1 + t < h:
                image[y1 + t, x1:x2] = color
            # Bottom
            if y2 - t >= 0:
                image[y2 - t, x1:x2] = color
            # Left
            if x1 + t < w:
                image[y1:y2, x1 + t] = color
            # Right
            if x2 - t >= 0:
                image[y1:y2, x2 - t] = color


def _draw_text(
    image: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
) -> None:
    """Draw text on image."""
    try:
        import cv2
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)
    except ImportError:
        # PIL fallback
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            pil_img = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_img)
            
            try:
                font = ImageFont.truetype("arial.ttf", int(font_scale * 24))
            except Exception:
                font = ImageFont.load_default()
            
            draw.text(pos, text, fill=color[::-1], font=font)
            image[:] = np.array(pil_img)
        except ImportError:
            pass


def save_faces(
    faces: List[np.ndarray],
    output_dir: str,
    prefix: str = "face",
    format: str = "jpg",
    start_index: int = 0,
) -> List[str]:
    """
    Save face images to files.
    
    Args:
        faces: List of face images.
        output_dir: Directory to save faces.
        prefix: Filename prefix.
        format: Image format (jpg, png).
        start_index: Starting index for numbering.
    
    Returns:
        List of saved file paths.
    """
    from PIL import Image
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, face in enumerate(faces):
        filename = f"{prefix}_{start_index + i:04d}.{format}"
        filepath = output_dir / filename
        
        pil_img = Image.fromarray(face)
        pil_img.save(filepath)
        saved_paths.append(str(filepath))
    
    return saved_paths


def load_face(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load a face image from file.
    
    Args:
        path: Path to image file.
        size: Resize to this size.
    
    Returns:
        Face image as numpy array.
    """
    from PIL import Image
    
    img = Image.open(path)
    
    if size:
        img = img.resize(size, Image.LANCZOS)
    
    return np.array(img)


def preprocess_face(
    face: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
    normalize: bool = True,
    grayscale: bool = False,
) -> np.ndarray:
    """
    Preprocess a face image for recognition.
    
    Args:
        face: Input face image.
        target_size: Target size (width, height).
        normalize: Normalize pixel values.
        grayscale: Convert to grayscale.
    
    Returns:
        Preprocessed face image.
    """
    from PIL import Image
    
    # Resize
    pil_img = Image.fromarray(face)
    pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    # Grayscale
    if grayscale:
        pil_img = pil_img.convert('L')
    
    result = np.array(pil_img).astype(np.float32)
    
    # Normalize
    if normalize:
        result = result / 255.0
        # Standardize
        mean = np.mean(result)
        std = np.std(result) + 1e-7
        result = (result - mean) / std
    
    return result


def compute_face_distance(
    face1: np.ndarray,
    face2: np.ndarray,
    method: str = "euclidean",
) -> float:
    """
    Compute distance between two face representations.
    
    Args:
        face1: First face embedding or image.
        face2: Second face embedding or image.
        method: Distance method ('euclidean', 'cosine', 'manhattan').
    
    Returns:
        Distance value.
    """
    f1 = face1.flatten().astype(np.float32)
    f2 = face2.flatten().astype(np.float32)
    
    if method == "euclidean":
        return float(np.sqrt(np.sum((f1 - f2) ** 2)))
    elif method == "cosine":
        return float(1 - np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-7))
    elif method == "manhattan":
        return float(np.sum(np.abs(f1 - f2)))
    else:
        raise ValueError(f"Unknown method: {method}")


def verify_faces(
    face1: np.ndarray,
    face2: np.ndarray,
    threshold: float = 0.6,
    method: str = "cosine",
) -> Tuple[bool, float]:
    """
    Verify if two faces belong to the same person.
    
    Args:
        face1: First face image.
        face2: Second face image.
        threshold: Distance threshold for matching.
        method: Distance method.
    
    Returns:
        (is_same_person, distance)
    """
    # Preprocess both faces
    f1 = preprocess_face(face1, grayscale=True)
    f2 = preprocess_face(face2, grayscale=True)
    
    distance = compute_face_distance(f1, f2, method=method)
    is_same = distance < threshold
    
    return is_same, distance


def augment_face(
    face: np.ndarray,
    flip: bool = True,
    rotate: bool = True,
    brightness: bool = True,
    contrast: bool = True,
    noise: bool = False,
) -> List[np.ndarray]:
    """
    Generate augmented versions of a face image.
    
    Args:
        face: Input face image.
        flip: Apply horizontal flip.
        rotate: Apply rotation.
        brightness: Apply brightness changes.
        contrast: Apply contrast changes.
        noise: Add noise.
    
    Returns:
        List of augmented face images.
    """
    augmented = [face]
    
    if flip:
        augmented.append(np.fliplr(face))
    
    if rotate:
        try:
            from scipy.ndimage import rotate as scipy_rotate
            for angle in [-10, 10]:
                rotated = scipy_rotate(face, angle, reshape=False, mode='constant')
                augmented.append(rotated.astype(np.uint8))
        except ImportError:
            pass
    
    if brightness:
        # Brighter
        brighter = np.clip(face.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        augmented.append(brighter)
        # Darker
        darker = np.clip(face.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        augmented.append(darker)
    
    if contrast:
        # Higher contrast
        mean = np.mean(face)
        high_contrast = np.clip((face.astype(np.float32) - mean) * 1.5 + mean, 0, 255).astype(np.uint8)
        augmented.append(high_contrast)
        # Lower contrast
        low_contrast = np.clip((face.astype(np.float32) - mean) * 0.5 + mean, 0, 255).astype(np.uint8)
        augmented.append(low_contrast)
    
    if noise:
        noisy = face.astype(np.float32) + np.random.normal(0, 10, face.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        augmented.append(noisy)
    
    return augmented


def resize_face(
    face: np.ndarray,
    size: Tuple[int, int],
    keep_aspect: bool = False,
) -> np.ndarray:
    """
    Resize a face image.
    
    Args:
        face: Input face image.
        size: Target size (width, height).
        keep_aspect: Keep aspect ratio and pad.
    
    Returns:
        Resized face image.
    """
    from PIL import Image
    
    pil_img = Image.fromarray(face)
    
    if keep_aspect:
        # Calculate new size maintaining aspect ratio
        aspect = pil_img.width / pil_img.height
        target_aspect = size[0] / size[1]
        
        if aspect > target_aspect:
            new_w = size[0]
            new_h = int(size[0] / aspect)
        else:
            new_h = size[1]
            new_w = int(size[1] * aspect)
        
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Pad
        result = Image.new(pil_img.mode, size, 0)
        offset = ((size[0] - new_w) // 2, (size[1] - new_h) // 2)
        result.paste(pil_img, offset)
        
        return np.array(result)
    
    pil_img = pil_img.resize(size, Image.LANCZOS)
    return np.array(pil_img)


def to_grayscale(face: np.ndarray) -> np.ndarray:
    """Convert face to grayscale."""
    if len(face.shape) == 2:
        return face
    return np.mean(face, axis=2).astype(np.uint8)


def to_rgb(face: np.ndarray) -> np.ndarray:
    """Convert face to RGB."""
    if len(face.shape) == 3 and face.shape[2] == 3:
        return face
    return np.stack([face] * 3, axis=-1)
