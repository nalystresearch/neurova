# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova hair region segmentation pipeline.

semantic segmentation of hair pixels for virtual try-on, color manipulation,
and photo editing applications.

output structure:
    masks: (h, w) float array with hair probability [0, 1]

applications:
    - hair color change: blend new color using mask alpha
    - hair style overlay: composite virtual hairstyles
    - beauty filters: selective adjustments to hair region

typical workflow:

    from neurova.solutions import HairSegmentation
    
    segmenter = HairSegmentation()
    
    with segmenter:
        out = segmenter.process(frame)
        
        if out.masks is not None:
            # change hair color to auburn
            colored = segmenter.apply_hair_color(
                frame, out.masks, color=(165, 42, 42), intensity=0.6
            )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from neurova.solutions.core import (
    NeuralPipeline,
    InferenceOutput,
    InferenceBackend,
    sigmoid,
)
from neurova.solutions.assets import (
    get_model_path,
    download_model,
)


class HairSegmentation(NeuralPipeline):
    """
    hair region segmentation pipeline.
    
    produces a mask indicating hair probability per pixel.
    best results with front-facing portrait images.
    
    parameters:
        threshold: binarization cutoff for hard masks
        feather_radius: edge softening amount (pixels)
    
    color utilities:
        apply_hair_color() - blend a solid color
        apply_hair_tint() - shift hue while preserving luminance
    
    example:
        hair = HairSegmentation()
        
        with hair:
            out = hair.process(frame)
            
            # apply purple tint
            colored = hair.apply_hair_color(
                frame, out.masks,
                color=(128, 0, 128),
                intensity=0.5
            )
    """
    
    NETWORK_DIM = 512
    
    def __init__(
        self,
        threshold: float = 0.5,
        feather_radius: int = 0,
        # legacy parameters
        min_detection_confidence: Optional[float] = None,
    ):
        """
        configure hair segmentation.
        
        args:
            threshold: probability cutoff for binarization
            feather_radius: gaussian blur radius for edge softening
        """
        if min_detection_confidence is not None:
            threshold = min_detection_confidence
        
        super().__init__(confidence_threshold=threshold)
        
        self.threshold = threshold
        self.feather_radius = feather_radius
    
    def _get_default_model_path(self) -> Path:
        """bundled model path."""
        return get_model_path("hair_segmentation")
    
    def _get_model_url(self) -> str:
        """remote model url."""
        from neurova.solutions.model_manager import MODEL_URLS
        return MODEL_URLS.get("hair_segmentation", "")
    
    def initialize(self) -> bool:
        """load hair segmentation model."""
        if self._initialized:
            return True
        
        model_path = self._get_default_model_path()
        
        if not model_path.exists():
            try:
                download_model("hair_segmentation")
                model_path = self._get_default_model_path()
            except Exception:
                return False
        
        if not self._load_interpreter(model_path):
            return False
        
        self._initialized = True
        return True
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """resize and normalize for network input."""
        from PIL import Image
        
        img = Image.fromarray(frame)
        img_resized = img.resize(
            (self.NETWORK_DIM, self.NETWORK_DIM),
            Image.BILINEAR
        )
        
        # convert to rgba (model expects 4 channels)
        img_rgba = img_resized.convert('RGBA')
        
        tensor = np.array(img_rgba, dtype=np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)
    
    def _upscale_mask(
        self,
        mask: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """resize mask to original dimensions."""
        from PIL import Image
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        mask_resized = mask_pil.resize((target_w, target_h), Image.BILINEAR)
        
        result = np.array(mask_resized, dtype=np.float32) / 255.0
        
        # optional feathering
        if self.feather_radius > 0:
            try:
                import cv2
                result = cv2.GaussianBlur(
                    result,
                    (self.feather_radius * 2 + 1, self.feather_radius * 2 + 1),
                    0
                )
            except ImportError:
                pass
        
        return result
    
    def process(self, frame: np.ndarray) -> InferenceOutput:
        """
        segment hair region from frame.
        
        args:
            frame: rgb image (h, w, 3) uint8
            
        returns:
            InferenceOutput with masks containing hair probability
        """
        if not self._initialized:
            if not self.initialize():
                return InferenceOutput(backend=InferenceBackend.NONE)
        
        if frame is None or frame.ndim < 3:
            return InferenceOutput(backend=self._backend)
        
        h, w = frame.shape[:2]
        
        tensor = self._preprocess_frame(frame)
        outputs = self._invoke(tensor)
        
        raw_mask = outputs[0].squeeze()
        
        if raw_mask.min() < 0 or raw_mask.max() > 1:
            raw_mask = sigmoid(raw_mask)
        
        mask = self._upscale_mask(raw_mask, h, w)
        
        return InferenceOutput(
            masks=mask,
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
    
    def apply_hair_color(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int],
        intensity: float = 0.5,
    ) -> np.ndarray:
        """
        blend a solid color onto hair region.
        
        args:
            frame: original rgb frame
            mask: hair probability mask
            color: rgb color tuple (0-255)
            intensity: blend strength (0-1)
            
        returns:
            frame with hair color applied
        """
        color_layer = np.full_like(frame, color, dtype=np.float32)
        mask_3d = mask[..., np.newaxis]
        
        # blend color into hair region
        blended = frame * (1 - mask_3d * intensity) + color_layer * mask_3d * intensity
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def apply_hair_tint(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        hue_shift: float,
        saturation_mult: float = 1.0,
    ) -> np.ndarray:
        """
        shift hair hue while preserving luminance.
        
        args:
            frame: original rgb frame
            mask: hair probability mask
            hue_shift: hue rotation in degrees (0-360)
            saturation_mult: saturation multiplier
            
        returns:
            frame with hair tint applied
        """
        try:
            import cv2
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # apply hue shift to hair region
            mask_2d = mask > self.threshold
            hsv[mask_2d, 0] = (hsv[mask_2d, 0] + hue_shift / 2) % 180
            hsv[mask_2d, 1] = np.clip(hsv[mask_2d, 1] * saturation_mult, 0, 255)
            
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            return result
        
        except ImportError:
            # cv2 not available, return original
            return frame
    
    def get_binary_mask(
        self,
        mask: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """convert probability mask to binary."""
        thresh = threshold if threshold is not None else self.threshold
        return ((mask > thresh) * 255).astype(np.uint8)


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
