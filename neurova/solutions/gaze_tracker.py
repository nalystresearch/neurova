# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova iris and gaze tracking pipeline.

detects eye regions and localizes iris position for gaze estimation,
attention tracking, and eye-based interaction.

output structure:
    left_iris: list of 5 Point3D (center + 4 cardinal points)
    right_iris: list of 5 Point3D
    left_eye_contour: 16-point eye boundary
    right_eye_contour: 16-point eye boundary

iris landmark layout (per eye):
    0: iris center
    1: iris left edge
    2: iris right edge
    3: iris top edge
    4: iris bottom edge

gaze estimation:
    the iris center relative to eye corners indicates gaze direction.
    gaze_x = (iris_center.x - eye_inner.x) / (eye_outer.x - eye_inner.x)
    gaze_y = (iris_center.y - eye_top.y) / (eye_bottom.y - eye_top.y)

typical workflow:

    from neurova.solutions import Iris
    
    tracker = Iris()
    
    with tracker:
        out = tracker.process(frame)
        
        if out.left_iris:
            gaze = tracker.estimate_gaze(out, 'left')
            print(f"looking {gaze}")  # 'left', 'right', 'center', 'up', 'down'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from neurova.solutions.core import (
    NeuralPipeline,
    InferenceOutput,
    Point3D,
    InferenceBackend,
    sigmoid,
)
from neurova.solutions.assets import (
    get_model_path,
    download_model,
)
from neurova.solutions.face_geometry import FaceMesh, VertexGroups


@dataclass
class GazeOutput:
    """
    combined output for iris and gaze tracking.
    
    attributes:
        left_iris: 5 iris keypoints for left eye
        right_iris: 5 iris keypoints for right eye
        left_eye_contour: 16-point eye boundary
        right_eye_contour: 16-point eye boundary
        frame_width: input frame width
        frame_height: input frame height
    """
    left_iris: List[Point3D] = field(default_factory=list)
    right_iris: List[Point3D] = field(default_factory=list)
    left_eye_contour: List[Point3D] = field(default_factory=list)
    right_eye_contour: List[Point3D] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    backend: InferenceBackend = InferenceBackend.NONE
    
    def __bool__(self) -> bool:
        """true if any iris detected."""
        return len(self.left_iris) > 0 or len(self.right_iris) > 0
    
    @property
    def left_iris_center(self) -> Optional[Point3D]:
        """center of left iris."""
        return self.left_iris[0] if self.left_iris else None
    
    @property
    def right_iris_center(self) -> Optional[Point3D]:
        """center of right iris."""
        return self.right_iris[0] if self.right_iris else None
    
    # legacy aliases
    @property
    def left_iris_landmarks(self) -> List[Point3D]:
        return self.left_iris
    
    @property
    def right_iris_landmarks(self) -> List[Point3D]:
        return self.right_iris
    
    @property
    def left_eye_landmarks(self) -> List[Point3D]:
        return self.left_eye_contour
    
    @property
    def right_eye_landmarks(self) -> List[Point3D]:
        return self.right_eye_contour


# legacy alias
IrisResult = GazeOutput


class Iris(NeuralPipeline):
    """
    iris localization and gaze estimation pipeline.
    
    uses face mesh to locate eye regions, then runs iris landmark
    model to precisely locate the iris within each eye.
    
    parameters:
        detection_threshold: face detector confidence cutoff
        gaze_sensitivity: threshold for gaze direction classification
    
    utilities:
        estimate_gaze() - classify gaze direction
        compute_gaze_vector() - normalized 2d gaze vector
        compute_eye_aspect_ratio() - blink detection metric
    
    example:
        iris = Iris()
        
        with iris:
            out = iris.process(frame)
            
            if out.left_iris_center:
                gaze = iris.estimate_gaze(out, 'left')
                print(f"gaze direction: {gaze}")
    """
    
    IRIS_INPUT_DIM = 64
    IRIS_POINTS = 5
    
    def __init__(
        self,
        detection_threshold: float = 0.5,
        gaze_sensitivity: float = 0.3,
        # legacy parameters
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
    ):
        """
        configure iris tracking.
        
        args:
            detection_threshold: face detection confidence
            gaze_sensitivity: threshold for gaze classification
        """
        if min_detection_confidence is not None:
            detection_threshold = min_detection_confidence
        
        super().__init__(confidence_threshold=detection_threshold)
        
        self.gaze_sensitivity = gaze_sensitivity
        
        # face mesh for eye localization
        self._face_mesh: Optional[FaceMesh] = None
    
    def _get_default_model_path(self) -> Path:
        """bundled iris model path."""
        return get_model_path("iris_landmark")
    
    def _get_model_url(self) -> str:
        """remote model url."""
        from neurova.solutions.model_manager import MODEL_URLS
        return MODEL_URLS.get("iris_landmark", "")
    
    def initialize(self) -> bool:
        """load face mesh and iris models."""
        if self._initialized:
            return True
        
        # face mesh for eye localization
        self._face_mesh = FaceMesh(
            max_faces=1,
            detect_threshold=self.min_detection_confidence,
        )
        if not self._face_mesh.initialize():
            return False
        
        # iris landmark model
        model_path = self._get_default_model_path()
        if not model_path.exists():
            try:
                download_model("iris_landmark")
                model_path = self._get_default_model_path()
            except Exception:
                return False
        
        if not self._load_interpreter(model_path):
            return False
        
        self._initialized = True
        return True
    
    def _extract_eye_roi(
        self,
        frame: np.ndarray,
        eye_vertices: List[Point3D],
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        crop eye region for iris model.
        
        returns roi tensor and transformation params.
        """
        h, w = frame.shape[:2]
        
        # compute bounding box from eye vertices
        xs = [v.x for v in eye_vertices]
        ys = [v.y for v in eye_vertices]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # expand roi
        pad = (x_max - x_min) * 0.5
        x_min = max(0, x_min - pad)
        x_max = min(1, x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(1, y_max + pad)
        
        # pixel coords
        px1 = int(x_min * w)
        py1 = int(y_min * h)
        px2 = int(x_max * w)
        py2 = int(y_max * h)
        
        roi = frame[py1:py2, px1:px2]
        
        # resize for iris model
        from PIL import Image
        pil_roi = Image.fromarray(roi)
        pil_resized = pil_roi.resize(
            (self.IRIS_INPUT_DIM, self.IRIS_INPUT_DIM),
            Image.BILINEAR
        )
        
        roi_tensor = np.array(pil_resized, dtype=np.float32) / 255.0
        
        transform = (
            px1 / w,
            py1 / h,
            (px2 - px1) / w,
            (py2 - py1) / h,
        )
        
        return roi_tensor, transform
    
    def _predict_iris(
        self,
        roi: np.ndarray,
        transform: Tuple[float, float, float, float],
    ) -> List[Point3D]:
        """predict 5 iris landmarks from eye roi."""
        if self._interpreter is None:
            return []
        
        tensor = np.expand_dims(roi, axis=0)
        outputs = self._invoke(tensor)
        
        raw_pts = outputs[0].reshape(-1, 3)
        x_off, y_off, roi_w, roi_h = transform
        
        iris = []
        for i in range(min(len(raw_pts), self.IRIS_POINTS)):
            p = raw_pts[i]
            
            x = (p[0] / self.IRIS_INPUT_DIM) * roi_w + x_off
            y = (p[1] / self.IRIS_INPUT_DIM) * roi_h + y_off
            z = p[2] / self.IRIS_INPUT_DIM
            
            iris.append(Point3D(
                x=float(np.clip(x, 0, 1)),
                y=float(np.clip(y, 0, 1)),
                z=float(z),
            ))
        
        return iris
    
    def process(self, frame: np.ndarray) -> GazeOutput:
        """
        detect eyes and localize irises.
        
        args:
            frame: rgb image (h, w, 3) uint8
            
        returns:
            GazeOutput with iris and eye contour keypoints
        """
        if not self._initialized:
            if not self.initialize():
                return GazeOutput(backend=InferenceBackend.NONE)
        
        if frame is None or frame.ndim < 3:
            return GazeOutput(backend=self._backend)
        
        h, w = frame.shape[:2]
        
        # get face mesh
        face_out = self._face_mesh.process(frame)
        
        if not face_out.keypoints:
            return GazeOutput(
                frame_width=w,
                frame_height=h,
                backend=self._backend,
            )
        
        face_verts = face_out.keypoints[0]
        
        # extract eye regions
        left_eye = self._face_mesh.extract_region(face_verts, 'orbital_left')
        right_eye = self._face_mesh.extract_region(face_verts, 'orbital_right')
        
        output = GazeOutput(
            left_eye_contour=left_eye,
            right_eye_contour=right_eye,
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
        
        # process left eye
        if left_eye:
            try:
                roi, transform = self._extract_eye_roi(frame, left_eye)
                output.left_iris = self._predict_iris(roi, transform)
            except Exception:
                pass
        
        # process right eye
        if right_eye:
            try:
                roi, transform = self._extract_eye_roi(frame, right_eye)
                output.right_iris = self._predict_iris(roi, transform)
            except Exception:
                pass
        
        return output
    
    def estimate_gaze(
        self,
        output: GazeOutput,
        eye: str = 'left',
    ) -> str:
        """
        classify gaze direction from iris position.
        
        args:
            output: GazeOutput from process()
            eye: 'left' or 'right'
            
        returns:
            direction string: 'center', 'left', 'right', 'up', 'down'
        """
        iris = output.left_iris if eye == 'left' else output.right_iris
        contour = output.left_eye_contour if eye == 'left' else output.right_eye_contour
        
        if not iris or not contour:
            return 'unknown'
        
        iris_center = iris[0]
        
        # compute eye bounds
        xs = [p.x for p in contour]
        ys = [p.y for p in contour]
        eye_center_x = (min(xs) + max(xs)) / 2
        eye_center_y = (min(ys) + max(ys)) / 2
        eye_width = max(xs) - min(xs)
        eye_height = max(ys) - min(ys)
        
        # relative offset
        dx = (iris_center.x - eye_center_x) / (eye_width + 1e-6)
        dy = (iris_center.y - eye_center_y) / (eye_height + 1e-6)
        
        thresh = self.gaze_sensitivity
        
        if abs(dx) < thresh and abs(dy) < thresh:
            return 'center'
        elif abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def compute_gaze_vector(
        self,
        output: GazeOutput,
        eye: str = 'left',
    ) -> Tuple[float, float]:
        """
        compute normalized 2d gaze vector.
        
        args:
            output: GazeOutput from process()
            eye: 'left' or 'right'
            
        returns:
            (gaze_x, gaze_y) in range [-1, 1]
        """
        iris = output.left_iris if eye == 'left' else output.right_iris
        contour = output.left_eye_contour if eye == 'left' else output.right_eye_contour
        
        if not iris or not contour:
            return (0.0, 0.0)
        
        iris_center = iris[0]
        
        xs = [p.x for p in contour]
        ys = [p.y for p in contour]
        eye_center_x = (min(xs) + max(xs)) / 2
        eye_center_y = (min(ys) + max(ys)) / 2
        eye_width = max(xs) - min(xs)
        eye_height = max(ys) - min(ys)
        
        gaze_x = 2 * (iris_center.x - eye_center_x) / (eye_width + 1e-6)
        gaze_y = 2 * (iris_center.y - eye_center_y) / (eye_height + 1e-6)
        
        return (float(np.clip(gaze_x, -1, 1)), float(np.clip(gaze_y, -1, 1)))
    
    def compute_eye_aspect_ratio(
        self,
        output: GazeOutput,
        eye: str = 'left',
    ) -> float:
        """
        compute eye aspect ratio for blink detection.
        
        ear = height / width, low values indicate closed eye.
        
        args:
            output: GazeOutput from process()
            eye: 'left' or 'right'
            
        returns:
            aspect ratio (typically 0.2-0.5 for open eye)
        """
        contour = output.left_eye_contour if eye == 'left' else output.right_eye_contour
        
        if len(contour) < 4:
            return 0.0
        
        xs = [p.x for p in contour]
        ys = [p.y for p in contour]
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        return height / (width + 1e-6)
    
    def close(self):
        """release resources."""
        super().close()
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None


# legacy index aliases
IRIS_CENTER = 0
IRIS_LEFT = 1
IRIS_RIGHT = 2
IRIS_TOP = 3
IRIS_BOTTOM = 4


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
