# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova unified body capture pipeline.

combines face geometry, hand articulation, and body kinematics into a single
inference pipeline for comprehensive full-body tracking applications.

output structure:
    - face_keypoints: 468 facial vertices
    - left_hand_keypoints: 21 left hand joints
    - right_hand_keypoints: 21 right hand joints
    - body_keypoints: 33 body pose keypoints
    - body_mask: optional segmentation mask

typical workflow:

    from neurova.solutions import Holistic
    
    tracker = Holistic()
    
    with tracker:
        result = tracker.process(frame)
        
        if result.face_keypoints:
            # facial expression analysis
            pass
        
        if result.left_hand_keypoints:
            # left hand gesture recognition
            pass
        
        if result.body_keypoints:
            # body pose analysis
            pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from neurova.solutions.core import (
    NeuralPipeline,
    InferenceOutput,
    Point3D,
    InferenceBackend,
)
from neurova.solutions.face_geometry import FaceMesh
from neurova.solutions.hand_tracker import Hands
from neurova.solutions.body_kinetics import Pose


@dataclass
class UnifiedBodyOutput:
    """
    combined output from face, hands, and pose pipelines.
    
    attributes:
        face_keypoints: 468 facial mesh vertices
        left_hand_keypoints: 21 left hand joints
        right_hand_keypoints: 21 right hand joints
        body_keypoints: 33 body pose keypoints
        body_mask: optional segmentation mask
        frame_width: input image width
        frame_height: input image height
        backend: inference backend used
    """
    face_keypoints: List[Point3D] = field(default_factory=list)
    left_hand_keypoints: List[Point3D] = field(default_factory=list)
    right_hand_keypoints: List[Point3D] = field(default_factory=list)
    body_keypoints: List[Point3D] = field(default_factory=list)
    body_mask: Optional[np.ndarray] = None
    frame_width: int = 0
    frame_height: int = 0
    backend: InferenceBackend = InferenceBackend.NONE
    
    def __bool__(self) -> bool:
        """true if any keypoints detected."""
        return (
            len(self.face_keypoints) > 0 or
            len(self.left_hand_keypoints) > 0 or
            len(self.right_hand_keypoints) > 0 or
            len(self.body_keypoints) > 0
        )
    
    # legacy property aliases
    @property
    def face_landmarks(self) -> List[Point3D]:
        return self.face_keypoints
    
    @property
    def left_hand_landmarks(self) -> List[Point3D]:
        return self.left_hand_keypoints
    
    @property
    def right_hand_landmarks(self) -> List[Point3D]:
        return self.right_hand_keypoints
    
    @property
    def pose_landmarks(self) -> List[Point3D]:
        return self.body_keypoints
    
    @property
    def segmentation_mask(self) -> Optional[np.ndarray]:
        return self.body_mask
    
    @property
    def image_width(self) -> int:
        return self.frame_width
    
    @property
    def image_height(self) -> int:
        return self.frame_height


# legacy alias
HolisticResult = UnifiedBodyOutput


class Holistic(NeuralPipeline):
    """
    unified full-body capture pipeline.
    
    orchestrates face, hands, and pose detection on the same frame,
    combining results into a single coherent output.
    
    parameters:
        detection_threshold: confidence cutoff for all detectors
        pose_variant: 'lite', 'full', or 'heavy' for body pose
        enable_face: run face mesh detection
        enable_hands: run hand detection
        enable_segmentation: output body segmentation mask
    
    example:
        holistic = Holistic(pose_variant='full')
        
        with holistic:
            out = holistic.process(frame)
            
            if out.face_keypoints:
                print(f"face: {len(out.face_keypoints)} vertices")
            if out.body_keypoints:
                print(f"body: {len(out.body_keypoints)} keypoints")
    """
    
    def __init__(
        self,
        detection_threshold: float = 0.5,
        pose_variant: str = 'full',
        enable_face: bool = True,
        enable_hands: bool = True,
        enable_segmentation: bool = False,
        # legacy parameters
        min_detection_confidence: Optional[float] = None,
        min_tracking_confidence: Optional[float] = None,
        model_complexity: Optional[int] = None,
        refine_face_landmarks: bool = False,
        static_image_mode: bool = False,
    ):
        """
        configure unified body capture.
        
        args:
            detection_threshold: confidence threshold for all detectors
            pose_variant: 'lite', 'full', or 'heavy'
            enable_face: whether to run face mesh
            enable_hands: whether to run hand detection
            enable_segmentation: output body mask
        """
        # legacy compatibility
        if min_detection_confidence is not None:
            detection_threshold = min_detection_confidence
        if model_complexity is not None:
            pose_variant = {0: 'lite', 1: 'full', 2: 'heavy'}.get(model_complexity, 'full')
        
        super().__init__(confidence_threshold=detection_threshold)
        
        self.pose_variant = pose_variant
        self.enable_face = enable_face
        self.enable_hands = enable_hands
        self.enable_segmentation = enable_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.static_image_mode = static_image_mode
        
        # legacy properties
        self.model_complexity = {'lite': 0, 'full': 1, 'heavy': 2}.get(pose_variant, 1)
        
        # component pipelines
        self._face_mesh: Optional[FaceMesh] = None
        self._hands: Optional[Hands] = None
        self._pose: Optional[Pose] = None
    
    def _get_default_model_path(self) -> Path:
        """primary model is pose."""
        from neurova.solutions.model_manager import get_model_path
        return get_model_path(f"pose_landmark_{self.pose_variant}")
    
    def _get_model_url(self) -> str:
        """pose model url."""
        from neurova.solutions.model_manager import MODEL_URLS
        return MODEL_URLS.get(f"pose_landmark_{self.pose_variant}", "")
    
    def initialize(self) -> bool:
        """
        initialize all component pipelines.
        
        pose is required; face and hands are optional.
        """
        if self._initialized:
            return True
        
        # pose is required
        self._pose = Pose(
            variant=self.pose_variant,
            detection_threshold=self.min_detection_confidence,
            segmentation=self.enable_segmentation,
            static_image_mode=self.static_image_mode,
        )
        
        if not self._pose.initialize():
            return False
        
        self._backend = self._pose._backend
        
        # face mesh (optional)
        if self.enable_face:
            try:
                self._face_mesh = FaceMesh(
                    max_faces=1,
                    detect_threshold=self.min_detection_confidence,
                    refine_landmarks=self.refine_face_landmarks,
                    static_image_mode=self.static_image_mode,
                )
                self._face_mesh.initialize()
            except Exception:
                self._face_mesh = None
        
        # hands (optional)
        if self.enable_hands:
            try:
                self._hands = Hands(
                    max_hands=2,
                    palm_threshold=self.min_detection_confidence,
                    static_image_mode=self.static_image_mode,
                )
                self._hands.initialize()
            except Exception:
                self._hands = None
        
        self._initialized = True
        return True
    
    def process(self, frame: np.ndarray) -> UnifiedBodyOutput:
        """
        run unified body capture on frame.
        
        args:
            frame: rgb image (h, w, 3) uint8
            
        returns:
            UnifiedBodyOutput with all detected keypoints
        """
        if not self._initialized:
            if not self.initialize():
                return UnifiedBodyOutput()
        
        if frame is None or frame.ndim < 3:
            return UnifiedBodyOutput()
        
        h, w = frame.shape[:2]
        
        output = UnifiedBodyOutput(
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
        
        # body pose
        if self._pose is not None:
            pose_out = self._pose.process(frame)
            if pose_out.keypoints:
                output.body_keypoints = pose_out.keypoints[0]
            if pose_out.masks is not None:
                output.body_mask = pose_out.masks
        
        # face mesh
        if self._face_mesh is not None:
            face_out = self._face_mesh.process(frame)
            if face_out.keypoints:
                output.face_keypoints = face_out.keypoints[0]
        
        # hands
        if self._hands is not None:
            hands_out = self._hands.process(frame)
            if hands_out.keypoints:
                for hand_kpts in hands_out.keypoints:
                    if not hand_kpts:
                        continue
                    
                    # assign to left/right based on wrist position
                    wrist = hand_kpts[0]
                    
                    # mirror view: right side of image = left hand
                    if wrist.x > 0.5:
                        if not output.left_hand_keypoints:
                            output.left_hand_keypoints = hand_kpts
                    else:
                        if not output.right_hand_keypoints:
                            output.right_hand_keypoints = hand_kpts
        
        return output
    
    def close(self):
        """release all component resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._hands is not None:
            self._hands.close()
        if self._pose is not None:
            self._pose.close()
        
        self._face_mesh = None
        self._hands = None
        self._pose = None
        super().close()


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
