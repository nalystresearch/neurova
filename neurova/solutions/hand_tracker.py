# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova hand articulation pipeline.

real-time detection and tracking of hand pose with 21 articulation points.
optimized for gesture recognition, sign language, and interactive applications.

skeletal topology:
    each hand forms a tree structure rooted at the wrist (node 0).
    five kinematic chains branch from the wrist, one per digit.

    wrist (0)
     thumb chain:  1 (cmc) -> 2 (mcp) -> 3 (ip) -> 4 (tip)
     index chain:  5 (mcp) -> 6 (pip) -> 7 (dip) -> 8 (tip)
     middle chain: 9 (mcp) -> 10 (pip) -> 11 (dip) -> 12 (tip)
     ring chain:   13 (mcp) -> 14 (pip) -> 15 (dip) -> 16 (tip)
     pinky chain:  17 (mcp) -> 18 (pip) -> 19 (dip) -> 20 (tip)

    inter-digit connections (palm web): 5-9, 9-13, 13-17

gesture primitives:
    - finger flexion: compare tip-y to pip-y
    - finger extension: tip beyond mcp
    - pinch: thumb-tip to index-tip distance
    - spread: inter-tip distances

typical workflow:

    from neurova.solutions import Hands
    
    tracker = Hands(max_hands=2)
    
    with tracker:
        result = tracker.process(frame)
        
        for hand in result.keypoints:
            if tracker.is_finger_extended(hand, 'index'):
                print("pointing gesture")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple, Dict
from enum import IntEnum
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
# skeletal structure definitions
# 

class JointIndex(IntEnum):
    """named indices for hand articulation points."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


class DigitChain(NamedTuple):
    """joint indices forming a finger kinematic chain."""
    name: str
    mcp: int
    pip: int
    dip: int
    tip: int


# digit definitions
DIGITS = {
    'thumb': DigitChain('thumb', 1, 2, 3, 4),
    'index': DigitChain('index', 5, 6, 7, 8),
    'middle': DigitChain('middle', 9, 10, 11, 12),
    'ring': DigitChain('ring', 13, 14, 15, 16),
    'pinky': DigitChain('pinky', 17, 18, 19, 20),
}

# bone connections for rendering
SKELETON_EDGES = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # palm web
    (5, 9), (9, 13), (13, 17),
]

# fingertip indices for quick access
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

# mcp indices (knuckles) for gesture analysis
KNUCKLE_INDICES = [1, 5, 9, 13, 17]


# 
# hand articulation pipeline
# 

class Hands(NeuralPipeline):
    """
    hand pose estimator with 21 3d articulation points.
    
    two-stage architecture:
    1. palm detector locates hand bounding regions
    2. landmark regressor predicts joint positions per roi
    
    supports multi-hand tracking with configurable limits.
    
    parameters:
        max_hands: maximum simultaneous hands (1-4)
        palm_threshold: palm detector confidence cutoff
        joint_smoothing: temporal filter strength for video
        roi_scale: expansion factor for detector-to-landmark handoff
    
    gesture utilities:
        is_finger_extended() - check if finger is straightened
        is_fist() - detect closed hand
        pinch_distance() - thumb-to-index distance
        compute_hand_openness() - overall extension metric
    
    example:
        hands = Hands(max_hands=2, joint_smoothing=0.3)
        
        with hands:
            out = hands.process(frame)
            
            for hand_idx, joints in enumerate(out.keypoints):
                if hands.is_fist(joints):
                    print(f"hand {hand_idx}: fist detected")
    """
    
    JOINT_COUNT = 21
    DETECTOR_DIM = 192
    LANDMARK_DIM = 224
    
    def __init__(
        self,
        max_hands: int = 2,
        palm_threshold: float = 0.5,
        joint_smoothing: float = 0.0,
        roi_scale: float = 2.0,
        # legacy parameter names
        max_num_hands: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        static_image_mode: bool = False,
    ):
        """
        configure hand articulation pipeline.
        
        args:
            max_hands: max hands to track (1-4)
            palm_threshold: detector confidence threshold
            joint_smoothing: temporal smoothing (0=off, 1=max)
            roi_scale: palm roi expansion for landmark stage
        """
        # legacy compatibility
        if max_num_hands is not None:
            max_hands = max_num_hands
        if min_detection_confidence is not None:
            palm_threshold = min_detection_confidence
        
        super().__init__(
            confidence_threshold=palm_threshold,
            smoothing_factor=joint_smoothing,
        )
        
        self.max_hands = min(max(1, max_hands), 4)
        self.roi_scale = roi_scale
        self.joint_smoothing = joint_smoothing
        self.static_image_mode = static_image_mode
        
        # legacy properties
        self.max_num_hands = self.max_hands
        
        # detector runtime
        self._detector = None
        self._detector_backend = InferenceBackend.NONE
        self._anchors: Optional[np.ndarray] = None
        
        # tracking state
        self._prev_joints: Optional[List[List[Point3D]]] = None
    
    def _get_default_model_path(self) -> Path:
        """bundled landmark model path."""
        return get_model_path("hand_landmark")
    
    def _get_model_url(self) -> str:
        """remote model url."""
        from neurova.solutions.model_manager import MODEL_URLS
        return MODEL_URLS.get("hand_landmark", "")
    
    def _get_detector_path(self) -> Path:
        """bundled palm detector path."""
        return get_model_path("palm_detector_lite")
    
    def initialize(self) -> bool:
        """load palm detector and landmark models."""
        if self._initialized:
            return True
        
        # ensure detector model
        det_path = self._get_detector_path()
        if not det_path.exists():
            try:
                download_model("palm_detector_lite")
                det_path = self._get_detector_path()
            except Exception:
                pass
        
        # ensure landmark model
        lm_path = self._get_default_model_path()
        if not lm_path.exists():
            try:
                download_model("hand_landmark")
                lm_path = self._get_default_model_path()
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
        
        # load landmark model
        if not self._load_interpreter(lm_path):
            return False
        
        # build anchors
        self._anchors = self._build_anchors()
        
        self._initialized = True
        return True
    
    def _build_anchors(self) -> np.ndarray:
        """generate ssd anchors for palm detector."""
        anchors = []
        strides = [8, 16, 16, 16]
        
        for stride in strides:
            grid = self.DETECTOR_DIM // stride
            for y in range(grid):
                for x in range(grid):
                    cx = (x + 0.5) / grid
                    cy = (y + 0.5) / grid
                    anchors.append([cy, cx, 1.0, 1.0])
                    anchors.append([cy, cx, 1.0, 1.0])
        
        return np.array(anchors, dtype=np.float32)
    
    def _run_palm_detector(self, frame: np.ndarray) -> List[BoundingBox]:
        """detect palm regions in frame."""
        if self._detector is None:
            # fallback: single detection covering most of frame
            return [BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8, score=1.0)]
        
        # preprocess
        from PIL import Image
        img = Image.fromarray(frame)
        img_resized = img.resize(
            (self.DETECTOR_DIM, self.DETECTOR_DIM),
            Image.BILINEAR
        )
        tensor = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        
        # inference
        det_inputs = self._detector.get_input_details()
        det_outputs = self._detector.get_output_details()
        
        self._detector.set_tensor(det_inputs[0]["index"], tensor)
        self._detector.invoke()
        
        raw_boxes = self._detector.get_tensor(det_outputs[0]["index"])
        raw_scores = self._detector.get_tensor(det_outputs[1]["index"])
        
        # decode
        boxes = self._decode_palms(raw_boxes, raw_scores)
        boxes = nms_boxes(boxes, iou_threshold=0.3)
        
        return boxes[:self.max_hands]
    
    def _decode_palms(
        self,
        raw_boxes: np.ndarray,
        raw_scores: np.ndarray,
    ) -> List[BoundingBox]:
        """decode palm detector outputs."""
        raw_boxes = raw_boxes.reshape(-1, 18)
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
            anchor = self._anchors[idx]
            
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
    
    def _extract_hand_roi(
        self,
        frame: np.ndarray,
        box: BoundingBox,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """crop and prepare hand roi for landmark model."""
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
    
    def _predict_joints(
        self,
        roi: np.ndarray,
        transform: Tuple[float, float, float, float],
    ) -> List[Point3D]:
        """predict 21 joint positions from hand roi."""
        if self._interpreter is None:
            return []
        
        tensor = np.expand_dims(roi, axis=0)
        outputs = self._invoke(tensor)
        
        raw_joints = outputs[0].reshape(-1, 3)
        x_off, y_off, roi_w, roi_h = transform
        
        joints = []
        for i in range(min(len(raw_joints), self.JOINT_COUNT)):
            j = raw_joints[i]
            
            x = (j[0] / self.LANDMARK_DIM) * roi_w + x_off
            y = (j[1] / self.LANDMARK_DIM) * roi_h + y_off
            z = j[2] / self.LANDMARK_DIM
            
            joints.append(Point3D(
                x=float(np.clip(x, 0, 1)),
                y=float(np.clip(y, 0, 1)),
                z=float(z),
            ))
        
        while len(joints) < self.JOINT_COUNT:
            joints.append(Point3D(x=0.5, y=0.5, z=0.0))
        
        return joints
    
    def process(self, frame: np.ndarray) -> InferenceOutput:
        """
        run hand detection and landmark estimation.
        
        args:
            frame: rgb image (h, w, 3) uint8
            
        returns:
            InferenceOutput with boxes and keypoints
        """
        if not self._initialized:
            if not self.initialize():
                return InferenceOutput(backend=InferenceBackend.NONE)
        
        if frame is None or frame.ndim < 3:
            return InferenceOutput(backend=self._backend)
        
        h, w = frame.shape[:2]
        
        # detect palms
        boxes = self._run_palm_detector(frame)
        
        if not boxes:
            self._prev_joints = None
            return InferenceOutput(
                frame_width=w,
                frame_height=h,
                backend=self._backend,
            )
        
        # predict joints per hand
        all_joints = []
        for box in boxes:
            roi, transform = self._extract_hand_roi(frame, box)
            joints = self._predict_joints(roi, transform)
            all_joints.append(joints)
        
        # temporal smoothing
        if self.joint_smoothing > 0 and not self.static_image_mode:
            all_joints = self._apply_smoothing(all_joints)
        
        self._prev_joints = all_joints
        
        return InferenceOutput(
            boxes=boxes,
            keypoints=all_joints,
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
    
    def _apply_smoothing(
        self,
        current: List[List[Point3D]],
    ) -> List[List[Point3D]]:
        """apply temporal smoothing to joint positions."""
        if self._prev_joints is None or len(current) != len(self._prev_joints):
            return current
        
        alpha = 1.0 - self.joint_smoothing
        return [
            smooth_keypoints(curr, prev, alpha)
            for curr, prev in zip(current, self._prev_joints)
        ]
    
    #
    # gesture analysis utilities
    #
    
    def is_finger_extended(
        self,
        joints: List[Point3D],
        finger: str,
    ) -> bool:
        """
        check if a finger is extended (straightened).
        
        args:
            joints: 21-joint list from process()
            finger: 'thumb', 'index', 'middle', 'ring', or 'pinky'
        
        returns:
            true if finger tip is beyond pip in direction from wrist
        """
        if finger not in DIGITS:
            return False
        
        digit = DIGITS[finger]
        wrist = joints[JointIndex.WRIST]
        mcp = joints[digit.mcp]
        tip = joints[digit.tip]
        
        # thumb uses different logic (lateral extension)
        if finger == 'thumb':
            # compare x-distance from wrist
            return abs(tip.x - wrist.x) > abs(mcp.x - wrist.x)
        
        # other fingers: tip should be above (lower y) than pip
        pip = joints[digit.pip]
        return tip.y < pip.y
    
    def is_fist(self, joints: List[Point3D], threshold: float = 0.1) -> bool:
        """
        check if hand is in fist position.
        
        all fingers (except thumb) should be curled.
        """
        for finger in ['index', 'middle', 'ring', 'pinky']:
            if self.is_finger_extended(joints, finger):
                return False
        return True
    
    def pinch_distance(self, joints: List[Point3D]) -> float:
        """
        compute normalized distance between thumb tip and index tip.
        
        useful for pinch gesture detection.
        """
        thumb_tip = joints[JointIndex.THUMB_TIP]
        index_tip = joints[JointIndex.INDEX_TIP]
        return thumb_tip.distance_to(index_tip)
    
    def compute_hand_openness(self, joints: List[Point3D]) -> float:
        """
        compute overall hand openness metric (0=closed, 1=open).
        
        based on average finger extension.
        """
        extended_count = sum(
            1 for f in ['index', 'middle', 'ring', 'pinky']
            if self.is_finger_extended(joints, f)
        )
        return extended_count / 4.0
    
    def get_fingertips(self, joints: List[Point3D]) -> List[Point3D]:
        """extract the 5 fingertip joints."""
        return [joints[i] for i in FINGERTIP_INDICES]
    
    def get_knuckles(self, joints: List[Point3D]) -> List[Point3D]:
        """extract the 5 knuckle (mcp) joints."""
        return [joints[i] for i in KNUCKLE_INDICES]
    
    def close(self):
        """release resources."""
        super().close()
        self._detector = None
        self._prev_joints = None


# legacy exports
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HAND_CONNECTIONS = SKELETON_EDGES
FINGER_TIPS = FINGERTIP_INDICES
FINGER_PIPS = [3, 6, 10, 14, 18]


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
