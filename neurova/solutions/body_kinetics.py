# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova body kinematics pipeline.

full-body pose estimation with 33 anatomical keypoints for motion analysis,
fitness tracking, sports performance, and interactive applications.

anatomical keypoint layout:

    head & face (0-10):
        0: nasal bridge (nose)
        1-6: orbital points (inner/center/outer for each eye)
        7-8: auricular points (ears)
        9-10: labial corners (mouth)
    
    upper body (11-22):
        11-12: glenohumeral joints (shoulders)
        13-14: cubital joints (elbows)
        15-16: radiocarpal joints (wrists)
        17-22: digital references (pinky/index/thumb per hand)
    
    lower body (23-32):
        23-24: coxofemoral joints (hips)
        25-26: femorotibial joints (knees)
        27-28: talocrural joints (ankles)
        29-30: calcaneal points (heels)
        31-32: metatarsal points (foot tips)

model variants:
    lite   - fastest, suitable for mobile/embedded
    full   - balanced speed/accuracy
    heavy  - highest accuracy, more compute

typical workflow:

    from neurova.solutions import Pose
    
    estimator = Pose(variant='full')
    
    with estimator:
        result = estimator.process(frame)
        
        if result.keypoints:
            skeleton = result.keypoints[0]
            left_knee = skeleton[25]
            right_knee = skeleton[26]
            
            knee_angle = estimator.compute_joint_angle(
                skeleton, 'left_knee'
            )
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple, Dict
from enum import IntEnum
import math
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
# anatomical keypoint definitions
# 

class BodyJoint(IntEnum):
    """named indices for body keypoints."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class LimbDefinition(NamedTuple):
    """defines a limb by its proximal and distal joints."""
    name: str
    proximal: int
    distal: int


class JointAngle(NamedTuple):
    """defines a joint angle by three keypoints."""
    name: str
    a: int  # first endpoint
    b: int  # vertex (joint center)
    c: int  # second endpoint


# skeletal connections for visualization
SKELETON_EDGES = [
    # face mesh
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # torso box
    (11, 12), (11, 23), (12, 24), (23, 24),
    # left arm chain
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # right arm chain
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # left leg chain
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # right leg chain
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# limb definitions for length calculations
LIMBS = {
    'left_upper_arm': LimbDefinition('left_upper_arm', 11, 13),
    'left_forearm': LimbDefinition('left_forearm', 13, 15),
    'right_upper_arm': LimbDefinition('right_upper_arm', 12, 14),
    'right_forearm': LimbDefinition('right_forearm', 14, 16),
    'left_thigh': LimbDefinition('left_thigh', 23, 25),
    'left_shin': LimbDefinition('left_shin', 25, 27),
    'right_thigh': LimbDefinition('right_thigh', 24, 26),
    'right_shin': LimbDefinition('right_shin', 26, 28),
    'torso': LimbDefinition('torso', 11, 23),  # left side
    'shoulder_span': LimbDefinition('shoulder_span', 11, 12),
    'hip_span': LimbDefinition('hip_span', 23, 24),
}

# joint angle definitions for biomechanical analysis
JOINT_ANGLES = {
    'left_elbow': JointAngle('left_elbow', 11, 13, 15),
    'right_elbow': JointAngle('right_elbow', 12, 14, 16),
    'left_shoulder': JointAngle('left_shoulder', 13, 11, 23),
    'right_shoulder': JointAngle('right_shoulder', 14, 12, 24),
    'left_hip': JointAngle('left_hip', 11, 23, 25),
    'right_hip': JointAngle('right_hip', 12, 24, 26),
    'left_knee': JointAngle('left_knee', 23, 25, 27),
    'right_knee': JointAngle('right_knee', 24, 26, 28),
    'left_ankle': JointAngle('left_ankle', 25, 27, 31),
    'right_ankle': JointAngle('right_ankle', 26, 28, 32),
}

# anatomical regions for partial visibility checks
BODY_REGIONS = {
    'face': list(range(0, 11)),
    'upper_body': [11, 12, 13, 14, 15, 16],
    'hands': [17, 18, 19, 20, 21, 22],
    'lower_body': [23, 24, 25, 26, 27, 28],
    'feet': [29, 30, 31, 32],
    'left_side': [1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
    'right_side': [4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
}


# 
# pose estimation pipeline
# 

class Pose(NeuralPipeline):
    """
    body pose estimator with 33 anatomical keypoints.
    
    two-stage architecture:
    1. person detector localizes body bounding boxes
    2. pose regressor predicts keypoint positions per roi
    
    model variants (via model_complexity or variant parameter):
        0 / 'lite'  - fast inference, lower accuracy
        1 / 'full'  - balanced (default)
        2 / 'heavy' - highest accuracy, slower
    
    parameters:
        variant: 'lite', 'full', or 'heavy' (or model_complexity 0-2)
        keypoint_smoothing: temporal filter strength for video
        detection_threshold: person detector confidence cutoff
        segmentation: whether to output body segmentation mask
    
    biomechanical utilities:
        compute_joint_angle() - angle at a joint in degrees
        compute_limb_length() - distance between joint endpoints
        get_body_region() - extract keypoints for anatomical region
        estimate_body_orientation() - facing direction estimate
    
    example:
        pose = Pose(variant='full', keypoint_smoothing=0.3)
        
        with pose:
            out = pose.process(frame)
            
            if out.keypoints:
                skeleton = out.keypoints[0]
                knee_angle = pose.compute_joint_angle(skeleton, 'left_knee')
                print(f"left knee: {knee_angle:.1f} degrees")
    """
    
    KEYPOINT_COUNT = 33
    DETECTOR_DIM = 224
    LANDMARK_DIM = 256
    
    MODEL_VARIANTS = {
        0: 'pose_landmark_lite',
        1: 'pose_landmark_full',
        2: 'pose_landmark_heavy',
        'lite': 'pose_landmark_lite',
        'full': 'pose_landmark_full',
        'heavy': 'pose_landmark_heavy',
    }
    
    def __init__(
        self,
        variant: str = 'full',
        keypoint_smoothing: float = 0.0,
        detection_threshold: float = 0.5,
        segmentation: bool = False,
        roi_scale: float = 1.25,
        # legacy parameters
        model_complexity: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        enable_segmentation: Optional[bool] = None,
        static_image_mode: bool = False,
    ):
        """
        configure pose estimation pipeline.
        
        args:
            variant: 'lite', 'full', or 'heavy'
            keypoint_smoothing: temporal smoothing (0=off, 1=max)
            detection_threshold: detector confidence threshold
            segmentation: output body segmentation mask
            roi_scale: roi expansion factor
        """
        # legacy compatibility
        if model_complexity is not None:
            variant = model_complexity
        if min_detection_confidence is not None:
            detection_threshold = min_detection_confidence
        if enable_segmentation is not None:
            segmentation = enable_segmentation
        
        super().__init__(
            confidence_threshold=detection_threshold,
            smoothing_factor=keypoint_smoothing,
        )
        
        self.variant = variant
        self.keypoint_smoothing = keypoint_smoothing
        self.segmentation = segmentation
        self.roi_scale = roi_scale
        self.static_image_mode = static_image_mode
        
        # legacy properties
        self.model_complexity = 1 if variant == 'full' else (0 if variant == 'lite' else 2)
        self.enable_segmentation = segmentation
        
        # detector runtime
        self._detector = None
        self._detector_backend = InferenceBackend.NONE
        self._anchors: Optional[np.ndarray] = None
        
        # tracking state
        self._prev_skeleton: Optional[List[List[Point3D]]] = None
    
    def _get_model_name(self) -> str:
        """resolve variant to model filename."""
        return self.MODEL_VARIANTS.get(self.variant, 'pose_landmark_full')
    
    def _get_default_model_path(self) -> Path:
        """bundled pose model path."""
        return get_model_path(self._get_model_name())
    
    def _get_model_url(self) -> str:
        """remote model url."""
        from neurova.solutions.model_manager import MODEL_URLS
        return MODEL_URLS.get(self._get_model_name(), "")
    
    def _get_detector_path(self) -> Path:
        """bundled person detector path."""
        return get_model_path("pose_detector")
    
    def initialize(self) -> bool:
        """load person detector and pose models."""
        if self._initialized:
            return True
        
        # ensure detector
        det_path = self._get_detector_path()
        if not det_path.exists():
            try:
                download_model("pose_detector")
                det_path = self._get_detector_path()
            except Exception:
                pass
        
        # ensure pose model
        pose_path = self._get_default_model_path()
        if not pose_path.exists():
            try:
                download_model(self._get_model_name())
                pose_path = self._get_default_model_path()
            except Exception:
                return False
        
        # load detector
        if det_path.exists():
            try:
                self._detector, self._detector_backend = RuntimeLoader.get_interpreter(
                    str(det_path)
                )
                self._detector.allocate_tensors()
            except Exception:
                self._detector = None
        
        # load pose model
        if not self._load_interpreter(pose_path):
            return False
        
        # build anchors
        self._anchors = self._build_anchors()
        
        self._initialized = True
        return True
    
    def _build_anchors(self) -> np.ndarray:
        """generate ssd anchors for person detector."""
        anchors = []
        strides = [8, 16, 32, 32, 32]
        
        for stride in strides:
            grid = self.DETECTOR_DIM // stride
            for y in range(grid):
                for x in range(grid):
                    cx = (x + 0.5) / grid
                    cy = (y + 0.5) / grid
                    anchors.append([cy, cx, 1.0, 1.0])
        
        return np.array(anchors, dtype=np.float32)
    
    def _run_person_detector(self, frame: np.ndarray) -> List[BoundingBox]:
        """detect person bounding boxes."""
        if self._detector is None:
            # fallback: assume person fills frame
            return [BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0, score=1.0)]
        
        from PIL import Image
        img = Image.fromarray(frame)
        img_resized = img.resize(
            (self.DETECTOR_DIM, self.DETECTOR_DIM),
            Image.BILINEAR
        )
        tensor = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        
        det_inputs = self._detector.get_input_details()
        det_outputs = self._detector.get_output_details()
        
        self._detector.set_tensor(det_inputs[0]["index"], tensor)
        self._detector.invoke()
        
        raw_boxes = self._detector.get_tensor(det_outputs[0]["index"])
        raw_scores = self._detector.get_tensor(det_outputs[1]["index"])
        
        boxes = self._decode_persons(raw_boxes, raw_scores)
        boxes = nms_boxes(boxes, iou_threshold=0.3)
        
        return boxes[:1]  # pose typically single person
    
    def _decode_persons(
        self,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
    ) -> List[BoundingBox]:
        """decode person detector outputs."""
        raw_boxes = raw_boxes.reshape(-1, 12)
        scores = sigmoid(raw_scores.reshape(-1))
        
        valid_mask = scores > self.min_detection_confidence
        valid_idx = np.nonzero(valid_mask)[0]
        
        if len(valid_idx) == 0:
            return []
        
        boxes = []
        for idx in valid_idx:
            if idx >= len(self._anchors):
                continue
            
            score = float(scores[idx])
            box = raw_boxes[idx]
            anchor = self._anchors[idx] if idx < len(self._anchors) else [0.5, 0.5, 1, 1]
            
            cx = box[0] / self.DETECTOR_DIM + anchor[1]
            cy = box[1] / self.DETECTOR_DIM + anchor[0]
            w = box[2] / self.DETECTOR_DIM
            h = box[3] / self.DETECTOR_DIM
            
            boxes.append(BoundingBox(
                x=float(np.clip(cx - w/2, 0, 1)),
                y=float(np.clip(cy - h/2, 0, 1)),
                width=float(np.clip(w, 0, 1)),
                height=float(np.clip(h, 0, 1)),
                score=score,
            ))
        
        return boxes
    
    def _extract_body_roi(
        self,
        frame: np.ndarray,
        box: BoundingBox,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """crop and prepare body roi for pose model."""
        h, w = frame.shape[:2]
        
        cx, cy = box.center
        span = max(box.width, box.height) * self.roi_scale
        
        px1 = max(0, int((cx - span/2) * w))
        py1 = max(0, int((cy - span/2) * h))
        px2 = min(w, int((cx + span/2) * w))
        py2 = min(h, int((cy + span/2) * h))
        
        roi_crop = frame[py1:py2, px1:px2]
        
        from PIL import Image
        pil_roi = Image.fromarray(roi_crop)
        pil_resized = pil_roi.resize(
            (self.LANDMARK_DIM, self.LANDMARK_DIM),
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
    
    def _predict_keypoints(
        self,
        roi: np.ndarray,
        transform: Tuple[float, float, float, float],
    ) -> Tuple[List[Point3D], Optional[np.ndarray]]:
        """predict 33 keypoints from body roi."""
        if self._interpreter is None:
            return [], None
        
        tensor = np.expand_dims(roi, axis=0)
        outputs = self._invoke(tensor)
        
        raw_kpts = outputs[0].reshape(-1, 5)  # x, y, z, visibility, presence
        x_off, y_off, roi_w, roi_h = transform
        
        keypoints = []
        for i in range(min(len(raw_kpts), self.KEYPOINT_COUNT)):
            k = raw_kpts[i]
            
            x = (k[0] / self.LANDMARK_DIM) * roi_w + x_off
            y = (k[1] / self.LANDMARK_DIM) * roi_h + y_off
            z = k[2] / self.LANDMARK_DIM
            vis = sigmoid(np.array([k[3]]))[0] if len(k) > 3 else 1.0
            
            keypoints.append(Point3D(
                x=float(np.clip(x, 0, 1)),
                y=float(np.clip(y, 0, 1)),
                z=float(z),
                confidence=float(vis),
                visible=vis > 0.5,
            ))
        
        while len(keypoints) < self.KEYPOINT_COUNT:
            keypoints.append(Point3D(x=0.5, y=0.5, z=0.0, confidence=0.0, visible=False))
        
        # segmentation mask if requested
        seg_mask = None
        if self.segmentation and len(outputs) > 1:
            seg_mask = sigmoid(outputs[1].squeeze())
        
        return keypoints, seg_mask
    
    def process(self, frame: np.ndarray) -> InferenceOutput:
        """
        run pose estimation on frame.
        
        args:
            frame: rgb image (h, w, 3) uint8
            
        returns:
            InferenceOutput with boxes, keypoints, and optional mask
        """
        if not self._initialized:
            if not self.initialize():
                return InferenceOutput(backend=InferenceBackend.NONE)
        
        if frame is None or frame.ndim < 3:
            return InferenceOutput(backend=self._backend)
        
        h, w = frame.shape[:2]
        
        # detect persons
        boxes = self._run_person_detector(frame)
        
        if not boxes:
            self._prev_skeleton = None
            return InferenceOutput(
                frame_width=w,
                frame_height=h,
                backend=self._backend,
            )
        
        # predict pose per person
        all_keypoints = []
        seg_mask = None
        
        for box in boxes:
            roi, transform = self._extract_body_roi(frame, box)
            kpts, mask = self._predict_keypoints(roi, transform)
            all_keypoints.append(kpts)
            if mask is not None:
                seg_mask = mask
        
        # temporal smoothing
        if self.keypoint_smoothing > 0 and not self.static_image_mode:
            all_keypoints = self._apply_smoothing(all_keypoints)
        
        self._prev_skeleton = all_keypoints
        
        return InferenceOutput(
            boxes=boxes,
            keypoints=all_keypoints,
            masks=seg_mask,
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
    
    def _apply_smoothing(
        self,
        current: List[List[Point3D]],
    ) -> List[List[Point3D]]:
        """apply temporal smoothing to keypoints."""
        if self._prev_skeleton is None or len(current) != len(self._prev_skeleton):
            return current
        
        alpha = 1.0 - self.keypoint_smoothing
        return [
            smooth_keypoints(curr, prev, alpha)
            for curr, prev in zip(current, self._prev_skeleton)
        ]
    
    #
    # biomechanical analysis utilities
    #
    
    def compute_joint_angle(
        self,
        skeleton: List[Point3D],
        joint_name: str,
    ) -> float:
        """
        compute angle at a named joint in degrees.
        
        args:
            skeleton: 33-keypoint list from process()
            joint_name: one of 'left_elbow', 'right_elbow', 'left_knee', etc.
        
        returns:
            angle in degrees (0-180)
        """
        if joint_name not in JOINT_ANGLES:
            return 0.0
        
        jdef = JOINT_ANGLES[joint_name]
        a = skeleton[jdef.a]
        b = skeleton[jdef.b]
        c = skeleton[jdef.c]
        
        # vectors from vertex to endpoints
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        # angle via dot product
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        
        return float(np.degrees(angle_rad))
    
    def compute_limb_length(
        self,
        skeleton: List[Point3D],
        limb_name: str,
    ) -> float:
        """
        compute normalized length of a limb segment.
        
        args:
            skeleton: 33-keypoint list
            limb_name: one of 'left_upper_arm', 'left_forearm', etc.
        
        returns:
            euclidean distance in normalized coordinates
        """
        if limb_name not in LIMBS:
            return 0.0
        
        limb = LIMBS[limb_name]
        p = skeleton[limb.proximal]
        d = skeleton[limb.distal]
        
        return p.distance_to(d)
    
    def get_body_region(
        self,
        skeleton: List[Point3D],
        region: str,
    ) -> List[Point3D]:
        """
        extract keypoints for an anatomical region.
        
        args:
            skeleton: 33-keypoint list
            region: 'face', 'upper_body', 'hands', 'lower_body', 'feet',
                   'left_side', or 'right_side'
        
        returns:
            list of keypoints for that region
        """
        indices = BODY_REGIONS.get(region, [])
        return [skeleton[i] for i in indices if i < len(skeleton)]
    
    def estimate_body_orientation(
        self,
        skeleton: List[Point3D],
    ) -> str:
        """
        estimate whether body is facing front, back, left, or right.
        
        based on relative positions of shoulders and hips.
        """
        ls = skeleton[BodyJoint.LEFT_SHOULDER]
        rs = skeleton[BodyJoint.RIGHT_SHOULDER]
        
        # shoulder width vs depth
        width = abs(ls.x - rs.x)
        depth = abs(ls.z - rs.z)
        
        if width > depth * 2:
            return 'front' if ls.x < rs.x else 'back'
        elif ls.z < rs.z:
            return 'left'
        else:
            return 'right'
    
    def close(self):
        """release resources."""
        super().close()
        self._detector = None
        self._prev_skeleton = None


# legacy exports for backward compatibility
POSE_LANDMARKS = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

POSE_CONNECTIONS = SKELETON_EDGES

FACE_LANDMARKS_IDX = BODY_REGIONS['face']
LEFT_ARM_IDX = [11, 13, 15, 17, 19, 21]
RIGHT_ARM_IDX = [12, 14, 16, 18, 20, 22]
LEFT_LEG_IDX = [23, 25, 27, 29, 31]
RIGHT_LEG_IDX = [24, 26, 28, 30, 32]
TORSO_IDX = [11, 12, 23, 24]


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
