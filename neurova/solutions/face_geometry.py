# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova face geometry pipeline.

dense facial landmark extraction using a cascaded detection-refinement approach.
outputs 468 3d vertices forming a triangulated face mesh suitable for:

    - expression capture and animation retargeting
    - face geometry reconstruction
    - gaze and attention tracking
    - cosmetic and ar filter placement
    - biometric feature extraction

architecture overview:
    stage 1: blazeface detector localizes face bounding boxes
    stage 2: landmark regressor predicts 468 vertices per roi

vertex groups (anatomical regions):
    ORBITAL_LEFT     - left eye and surrounding tissue (16 pts)
    ORBITAL_RIGHT    - right eye and surrounding tissue (16 pts)
    BROW_LEFT        - left eyebrow arch (8 pts)
    BROW_RIGHT       - right eyebrow arch (8 pts)  
    NASAL            - nose bridge and tip (4 pts)
    LABIAL_OUTER     - outer lip contour (20 pts)
    LABIAL_INNER     - inner lip boundary (20 pts)
    MANDIBLE         - face oval / jawline (36 pts)

typical workflow:
    
    from neurova.solutions import FaceMesh
    
    pipeline = FaceMesh(max_faces=2)
    
    with pipeline:
        result = pipeline.process(rgb_frame)
        
        if result.keypoints:
            vertices = result.keypoints_as_array(0)  # (468, 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple
import numpy as np

from neurova.solutions.core import (
    NeuralPipeline,
    InferenceOutput,
    Point3D,
    BoundingBox,
    RuntimeLoader,
    InferenceBackend,
    sigmoid,
    nms_boxes,
    smooth_keypoints,
)
from neurova.solutions.assets import (
    get_model_path,
    download_model,
)


# 
# vertex index groups for anatomical regions
# 

class VertexGroups:
    """
    named index collections for face mesh regions.
    
    use these to extract specific facial features from the full 468-vertex mesh.
    """
    
    # jawline and face boundary
    MANDIBLE = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    # left periorbital region
    ORBITAL_LEFT = [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
        386, 385, 384, 398
    ]
    
    # right periorbital region  
    ORBITAL_RIGHT = [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
        159, 160, 161, 246
    ]
    
    # left supraorbital (brow)
    BROW_LEFT = [336, 296, 334, 293, 300, 276, 283, 282]
    
    # right supraorbital (brow)
    BROW_RIGHT = [70, 63, 105, 66, 107, 55, 65, 52]
    
    # outer vermilion border
    LABIAL_OUTER = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
        270, 269, 267, 0, 37, 39, 40, 185
    ]
    
    # inner oral boundary
    LABIAL_INNER = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
        310, 311, 312, 13, 82, 81, 80, 191
    ]
    
    # nose tip and columella
    NASAL = [1, 2, 98, 327]
    
    # key anchor points for alignment
    ANCHORS = [1, 33, 133, 362, 263]  # nose, left_eye, right_eye corners


class AnchorConfig(NamedTuple):
    """configuration for ssd anchor generation."""
    strides: Tuple[int, ...] = (8, 16, 16, 16)
    aspect_ratios: Tuple[float, ...] = (1.0,)
    anchors_per_cell: int = 2


# 
# face mesh pipeline
# 

class FaceMesh(NeuralPipeline):
    """
    dense face geometry estimator with 468 3d vertices.
    
    implements a two-stage cascade:
    1. blazeface detector finds face regions
    2. regression network predicts vertex positions
    
    vertex coordinates are normalized to [0,1] relative to image dimensions.
    z-values encode relative depth (orthographic projection).
    
    parameters:
        max_faces: upper bound on simultaneous face detections
        roi_expansion: scale factor for detector-to-landmark roi handoff
        vertex_smoothing: temporal filter strength for video (0=off)
        detect_threshold: detector confidence cutoff
        landmark_threshold: per-vertex visibility threshold
    
    example:
        mesh = FaceMesh(max_faces=2, vertex_smoothing=0.5)
        
        with mesh:
            for frame in video:
                out = mesh.process(frame)
                
                for face_idx, verts in enumerate(out.keypoints):
                    # verts is list of 468 Point3D
                    jaw = mesh.extract_region(verts, 'mandible')
    """
    
    VERTEX_COUNT = 468
    LANDMARK_INPUT_DIM = 192
    DETECTOR_INPUT_DIM = 128
    
    def __init__(
        self,
        max_faces: int = 1,
        roi_expansion: float = 1.5,
        vertex_smoothing: float = 0.0,
        detect_threshold: float = 0.5,
        landmark_threshold: float = 0.5,
        # legacy aliases
        max_num_faces: Optional[int] = None,
        refine_landmarks: bool = False,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        static_image_mode: bool = False,
    ):
        """
        configure face mesh pipeline.
        
        args:
            max_faces: max faces to track (1-10)
            roi_expansion: face roi scale factor for landmark stage
            vertex_smoothing: temporal smoothing alpha (0=none, 1=max)
            detect_threshold: face detector confidence threshold
            landmark_threshold: vertex visibility threshold
        """
        # handle legacy parameter names for backward compatibility
        if max_num_faces is not None:
            max_faces = max_num_faces
        if min_detection_confidence is not None:
            detect_threshold = min_detection_confidence
        
        super().__init__(
            confidence_threshold=detect_threshold,
            smoothing_factor=vertex_smoothing,
        )
        
        self.max_faces = min(max(1, max_faces), 10)
        self.roi_expansion = roi_expansion
        self.vertex_smoothing = vertex_smoothing
        self.landmark_threshold = landmark_threshold
        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        
        # runtime components
        self._detector = None
        self._detector_backend = InferenceBackend.NONE
        self._anchor_config = AnchorConfig()
        self._anchors: Optional[np.ndarray] = None
        
        # temporal state
        self._prev_vertices: Optional[List[List[Point3D]]] = None
    
    def _get_default_model_path(self) -> Path:
        """bundled landmark model location."""
        return get_model_path("face_landmark_lite")
    
    def _get_model_url(self) -> str:
        """remote landmark model url for download."""
        from neurova.solutions.model_manager import MODEL_URLS
        return MODEL_URLS.get("face_landmark_lite", "")
    
    def _get_detector_path(self) -> Path:
        """bundled detector model location."""
        return get_model_path("blaze_face_short_range")
    
    def initialize(self) -> bool:
        """
        load detector and landmark models.
        
        downloads models if not present. allocates inference runtimes.
        """
        if self._initialized:
            return True
        
        # ensure detector model exists
        detector_path = self._get_detector_path()
        if not detector_path.exists():
            try:
                download_model("blaze_face_short_range")
                detector_path = self._get_detector_path()
            except Exception:
                pass
        
        # ensure landmark model exists
        landmark_path = self._get_default_model_path()
        if not landmark_path.exists():
            try:
                download_model("face_landmark_lite")
                landmark_path = self._get_default_model_path()
            except Exception:
                return False
        
        # load detector runtime
        if detector_path.exists():
            try:
                self._detector, self._detector_backend = RuntimeLoader.get_interpreter(
                    str(detector_path)
                )
                self._detector.allocate_tensors()
            except Exception:
                self._detector = None
        
        # load landmark runtime
        if not self._load_interpreter(landmark_path):
            return False
        
        # precompute anchors
        self._anchors = self._build_anchors()
        
        self._initialized = True
        return True
    
    def _build_anchors(self) -> np.ndarray:
        """
        generate ssd-style anchor grid for detector decoding.
        
        anchors are (cy, cx, 1, 1) normalized coordinates.
        """
        anchors = []
        cfg = self._anchor_config
        
        for stride in cfg.strides:
            grid_dim = self.DETECTOR_INPUT_DIM // stride
            
            for row in range(grid_dim):
                for col in range(grid_dim):
                    cx = (col + 0.5) / grid_dim
                    cy = (row + 0.5) / grid_dim
                    
                    for _ in range(cfg.anchors_per_cell):
                        anchors.append([cy, cx, 1.0, 1.0])
        
        return np.array(anchors, dtype=np.float32)
    
    def _preprocess_detector(self, frame: np.ndarray) -> np.ndarray:
        """
        prepare frame for face detector.
        
        resizes to detector input size, normalizes to [0,1].
        """
        from PIL import Image
        
        img = Image.fromarray(frame)
        img_resized = img.resize(
            (self.DETECTOR_INPUT_DIM, self.DETECTOR_INPUT_DIM),
            Image.BILINEAR
        )
        
        tensor = np.array(img_resized, dtype=np.float32)
        tensor /= 255.0
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def _run_detector(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        execute face detection stage.
        
        returns list of face bounding boxes with scores.
        """
        if self._detector is None:
            return []
        
        # preprocess
        tensor = self._preprocess_detector(frame)
        
        # inference
        det_inputs = self._detector.get_input_details()
        det_outputs = self._detector.get_output_details()
        
        self._detector.set_tensor(det_inputs[0]["index"], tensor)
        self._detector.invoke()
        
        raw_boxes = self._detector.get_tensor(det_outputs[0]["index"])
        raw_scores = self._detector.get_tensor(det_outputs[1]["index"])
        
        # decode
        boxes = self._decode_boxes(raw_boxes, raw_scores)
        
        # nms and limit
        boxes = nms_boxes(boxes, iou_threshold=0.3)
        
        return boxes[:self.max_faces]
    
    def _decode_boxes(
        self,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
    ) -> List[BoundingBox]:
        """
        decode detector outputs to bounding boxes.
        
        applies anchor offsets and sigmoid to scores.
        """
        raw_boxes = raw_boxes.reshape(-1, 16)
        scores = sigmoid(raw_scores.reshape(-1))
        
        # confidence filter
        valid_mask = scores > self.min_detection_confidence
        valid_idx = np.nonzero(valid_mask)[0]
        
        if len(valid_idx) == 0:
            return []
        
        boxes = []
        for idx in valid_idx:
            score = float(scores[idx])
            box = raw_boxes[idx]
            anchor = self._anchors[idx]
            
            # decode center and size
            cx = box[0] / self.DETECTOR_INPUT_DIM + anchor[1]
            cy = box[1] / self.DETECTOR_INPUT_DIM + anchor[0]
            w = box[2] / self.DETECTOR_INPUT_DIM
            h = box[3] / self.DETECTOR_INPUT_DIM
            
            # convert to corner format
            x = cx - w / 2
            y = cy - h / 2
            
            boxes.append(BoundingBox(
                x=float(x),
                y=float(y),
                width=float(w),
                height=float(h),
                score=score,
            ))
        
        return boxes
    
    def _extract_roi(
        self,
        frame: np.ndarray,
        box: BoundingBox,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """
        crop and prepare face roi for landmark model.
        
        expands box by roi_expansion factor, handles boundary clipping.
        
        returns:
            (roi_tensor, (x_off, y_off, roi_w, roi_h)) normalized transform
        """
        h, w = frame.shape[:2]
        
        # expand roi around box center
        cx, cy = box.center
        span = max(box.width, box.height) * self.roi_expansion
        
        # pixel coordinates
        px1 = int((cx - span / 2) * w)
        py1 = int((cy - span / 2) * h)
        px2 = int((cx + span / 2) * w)
        py2 = int((cy + span / 2) * h)
        
        # clamp to frame
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(w, px2)
        py2 = min(h, py2)
        
        roi_crop = frame[py1:py2, px1:px2]
        
        # resize for landmark model
        from PIL import Image
        pil_roi = Image.fromarray(roi_crop)
        pil_resized = pil_roi.resize(
            (self.LANDMARK_INPUT_DIM, self.LANDMARK_INPUT_DIM),
            Image.BILINEAR
        )
        
        roi_tensor = np.array(pil_resized, dtype=np.float32) / 255.0
        
        # normalized transform for mapping back
        transform = (
            px1 / w,  # x offset
            py1 / h,  # y offset
            (px2 - px1) / w,  # roi width
            (py2 - py1) / h,  # roi height
        )
        
        return roi_tensor, transform
    
    def _predict_vertices(
        self,
        roi_tensor: np.ndarray,
        transform: Tuple[float, float, float, float],
    ) -> List[Point3D]:
        """
        run landmark model on roi, transform to image coordinates.
        
        returns 468 Point3D vertices.
        """
        if self._interpreter is None:
            return []
        
        # batch dimension
        input_tensor = np.expand_dims(roi_tensor, axis=0)
        
        # inference
        outputs = self._invoke(input_tensor)
        raw_verts = outputs[0].reshape(-1, 3)
        
        # transform to image coordinates
        x_off, y_off, roi_w, roi_h = transform
        
        vertices = []
        for i in range(min(len(raw_verts), self.VERTEX_COUNT)):
            v = raw_verts[i]
            
            # roi-local to image-normalized
            x = (v[0] / self.LANDMARK_INPUT_DIM) * roi_w + x_off
            y = (v[1] / self.LANDMARK_INPUT_DIM) * roi_h + y_off
            z = v[2] / self.LANDMARK_INPUT_DIM
            
            vertices.append(Point3D(
                x=float(np.clip(x, 0, 1)),
                y=float(np.clip(y, 0, 1)),
                z=float(z),
            ))
        
        # pad to full count if needed
        while len(vertices) < self.VERTEX_COUNT:
            vertices.append(Point3D(x=0.5, y=0.5, z=0.0))
        
        return vertices
    
    def process(self, frame: np.ndarray) -> InferenceOutput:
        """
        run full face mesh pipeline on frame.
        
        args:
            frame: rgb image as (h, w, 3) uint8 array
            
        returns:
            InferenceOutput with boxes and keypoints lists
        """
        # lazy init
        if not self._initialized:
            if not self.initialize():
                return InferenceOutput(backend=InferenceBackend.NONE)
        
        if frame is None or frame.ndim < 3:
            return InferenceOutput(backend=self._backend)
        
        h, w = frame.shape[:2]
        
        # stage 1: detect faces
        boxes = self._run_detector(frame)
        
        if not boxes:
            self._prev_vertices = None
            return InferenceOutput(
                frame_width=w,
                frame_height=h,
                backend=self._backend,
            )
        
        # stage 2: landmark regression per face
        all_vertices = []
        for box in boxes:
            roi, transform = self._extract_roi(frame, box)
            vertices = self._predict_vertices(roi, transform)
            all_vertices.append(vertices)
        
        # optional temporal smoothing
        if self.vertex_smoothing > 0 and not self.static_image_mode:
            all_vertices = self._apply_smoothing(all_vertices)
        
        self._prev_vertices = all_vertices
        
        return InferenceOutput(
            boxes=boxes,
            keypoints=all_vertices,
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
    
    def _apply_smoothing(
        self,
        current: List[List[Point3D]],
    ) -> List[List[Point3D]]:
        """apply temporal smoothing to vertex positions."""
        if self._prev_vertices is None:
            return current
        
        if len(current) != len(self._prev_vertices):
            return current
        
        alpha = 1.0 - self.vertex_smoothing
        smoothed = []
        
        for curr_face, prev_face in zip(current, self._prev_vertices):
            smoothed_face = smooth_keypoints(curr_face, prev_face, alpha)
            smoothed.append(smoothed_face)
        
        return smoothed
    
    #
    # region extraction utilities
    #
    
    def extract_region(
        self,
        vertices: List[Point3D],
        region: str,
    ) -> List[Point3D]:
        """
        extract vertices for a named anatomical region.
        
        args:
            vertices: full 468-vertex list
            region: one of 'mandible', 'orbital_left', 'orbital_right',
                   'brow_left', 'brow_right', 'labial_outer', 
                   'labial_inner', 'nasal', 'anchors'
        
        returns:
            subset of vertices for the region
        """
        region_map = {
            'mandible': VertexGroups.MANDIBLE,
            'orbital_left': VertexGroups.ORBITAL_LEFT,
            'orbital_right': VertexGroups.ORBITAL_RIGHT,
            'brow_left': VertexGroups.BROW_LEFT,
            'brow_right': VertexGroups.BROW_RIGHT,
            'labial_outer': VertexGroups.LABIAL_OUTER,
            'labial_inner': VertexGroups.LABIAL_INNER,
            'nasal': VertexGroups.NASAL,
            'anchors': VertexGroups.ANCHORS,
        }
        
        indices = region_map.get(region.lower(), [])
        return [vertices[i] for i in indices if i < len(vertices)]
    
    def get_face_oval(self, vertices: List[Point3D]) -> List[Point3D]:
        """legacy method: extract jawline vertices."""
        return self.extract_region(vertices, 'mandible')
    
    def get_left_eye(self, vertices: List[Point3D]) -> List[Point3D]:
        """legacy method: extract left eye vertices."""
        return self.extract_region(vertices, 'orbital_left')
    
    def get_right_eye(self, vertices: List[Point3D]) -> List[Point3D]:
        """legacy method: extract right eye vertices."""
        return self.extract_region(vertices, 'orbital_right')
    
    def get_lips(
        self,
        vertices: List[Point3D],
    ) -> Tuple[List[Point3D], List[Point3D]]:
        """legacy method: extract outer and inner lip vertices."""
        outer = self.extract_region(vertices, 'labial_outer')
        inner = self.extract_region(vertices, 'labial_inner')
        return outer, inner
    
    def close(self):
        """release all resources."""
        super().close()
        self._detector = None
        self._prev_vertices = None


# legacy index exports for backward compatibility
FACE_OVAL = VertexGroups.MANDIBLE
LEFT_EYE = VertexGroups.ORBITAL_LEFT
RIGHT_EYE = VertexGroups.ORBITAL_RIGHT
LEFT_EYEBROW = VertexGroups.BROW_LEFT
RIGHT_EYEBROW = VertexGroups.BROW_RIGHT
LIPS_OUTER = VertexGroups.LABIAL_OUTER
LIPS_INNER = VertexGroups.LABIAL_INNER
NOSE_TIP = VertexGroups.NASAL


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
