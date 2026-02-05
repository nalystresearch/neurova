# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova solutions module.

provides high level apis for common computer vision tasks using tflite models.
all solutions work offline with bundled models and automatic download fallback.

available solutions:
    - face_mesh: 468 3d face landmarks
    - hands: 21 hand landmarks per hand
    - pose: 33 body pose landmarks
    - holistic: combined face, hands, and pose
    - selfie_segmentation: background removal
    - hair_segmentation: hair masking
    - iris: eye and iris tracking

usage:
    from neurova.solutions import FaceMesh, Hands, Pose

    # face mesh with 468 landmarks
    face_mesh = FaceMesh()
    landmarks = face_mesh.process(image)

    # hand detection with 21 landmarks
    hands = Hands()
    results = hands.process(image)

    # body pose with 33 landmarks
    pose = Pose()
    landmarks = pose.process(image)
"""

from neurova.solutions.core import (
    # new primary classes
    NeuralPipeline,
    InferenceOutput,
    Point3D,
    BoundingBox,
    InferenceBackend,
    RuntimeLoader,
    # utilities
    sigmoid,
    softmax,
    nms_boxes,
    smooth_keypoints,
    # legacy aliases
    Solution,
    SolutionResult,
    Landmark,
    NormalizedLandmark,
    Detection,
)

from neurova.solutions.assets import (
    ModelManager,
    download_model,
    get_model_path,
    MODEL_URLS,
)

from neurova.solutions.face_geometry import FaceMesh, VertexGroups
from neurova.solutions.hand_tracker import Hands, JointIndex, DIGITS, SKELETON_EDGES
from neurova.solutions.body_kinetics import Pose, BodyJoint, LIMBS, JOINT_ANGLES
from neurova.solutions.unified_capture import Holistic, UnifiedBodyOutput
from neurova.solutions.person_segment import SelfieSegmentation
from neurova.solutions.hair_segment import HairSegmentation
from neurova.solutions.gaze_tracker import Iris, GazeOutput

__all__ = [
    # primary classes
    "NeuralPipeline",
    "InferenceOutput",
    "Point3D",
    "BoundingBox",
    "InferenceBackend",
    "RuntimeLoader",
    # utilities
    "sigmoid",
    "softmax",
    "nms_boxes",
    "smooth_keypoints",
    # legacy aliases
    "Solution",
    "SolutionResult",
    "Landmark",
    "NormalizedLandmark",
    "Detection",
    # model management
    "ModelManager",
    "download_model",
    "get_model_path",
    "MODEL_URLS",
    # solutions
    "FaceMesh",
    "VertexGroups",
    "Hands",
    "JointIndex",
    "DIGITS",
    "SKELETON_EDGES",
    "Pose",
    "BodyJoint",
    "LIMBS",
    "JOINT_ANGLES",
    "Holistic",
    "UnifiedBodyOutput",
    "SelfieSegmentation",
    "HairSegmentation",
    "Iris",
    "GazeOutput",
]

# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
