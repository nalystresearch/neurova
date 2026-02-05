# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova person segmentation pipeline.

semantic segmentation of person silhouette from background.
produces a probability mask suitable for background replacement,
virtual backgrounds, portrait effects, and video compositing.

output structure:
    masks: (h, w) float array with person probability [0, 1]
    
mask applications:
    - background blur: blur * (1 - mask) + frame * mask
    - background replace: new_bg * (1 - mask) + frame * mask
    - edge feathering: gaussian_blur(mask, sigma) for soft edges

typical workflow:

    from neurova.solutions import SelfieSegmentation
    
    segmenter = SelfieSegmentation()
    
    with segmenter:
        out = segmenter.process(frame)
        
        if out.masks is not None:
            person_mask = out.masks
            blurred_bg = gaussian_blur(frame, 15)
            composite = frame * mask[..., None] + blurred_bg * (1 - mask[..., None])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


class SelfieSegmentation(NeuralPipeline):
    """
    person silhouette segmentation pipeline.
    
    produces a single-channel mask indicating person probability per pixel.
    optimized for portrait and video call scenarios.
    
    model variants:
        'general' (0): works with various poses and distances
        'landscape' (1): optimized for landscape orientation
    
    parameters:
        variant: 'general' or 'landscape'
        threshold: binarization threshold for hard masks
        upscale_method: 'bilinear' or 'nearest' for mask resize
    
    example:
        seg = SelfieSegmentation(variant='general')
        
        with seg:
            out = seg.process(frame)
            
            mask = out.masks  # (h, w) float32
            
            # soft composite with blurred background
            import cv2
            blurred = cv2.GaussianBlur(frame, (21, 21), 0)
            composite = (frame * mask[..., None] + 
                        blurred * (1 - mask[..., None])).astype(np.uint8)
    """
    
    NETWORK_DIM = 256
    
    def __init__(
        self,
        variant: str = 'general',
        threshold: float = 0.5,
        upscale_method: str = 'bilinear',
        # legacy parameters
        model_selection: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
    ):
        """
        configure person segmentation.
        
        args:
            variant: 'general' or 'landscape'
            threshold: probability threshold for binarization
            upscale_method: 'bilinear' or 'nearest'
        """
        # legacy compatibility
        if model_selection is not None:
            variant = 'landscape' if model_selection == 1 else 'general'
        if min_detection_confidence is not None:
            threshold = min_detection_confidence
        
        super().__init__(confidence_threshold=threshold)
        
        self.variant = variant
        self.threshold = threshold
        self.upscale_method = upscale_method
        
        # legacy property
        self.model_selection = 1 if variant == 'landscape' else 0
    
    def _get_default_model_path(self) -> Path:
        """bundled segmentation model path."""
        name = "selfie_segmentation_landscape" if self.variant == 'landscape' else "selfie_segmentation"
        return get_model_path(name)
    
    def _get_model_url(self) -> str:
        """remote model url."""
        from neurova.solutions.model_manager import MODEL_URLS
        name = "selfie_segmentation_landscape" if self.variant == 'landscape' else "selfie_segmentation"
        return MODEL_URLS.get(name, "")
    
    def initialize(self) -> bool:
        """load segmentation model."""
        if self._initialized:
            return True
        
        model_name = "selfie_segmentation_landscape" if self.variant == 'landscape' else "selfie_segmentation"
        model_path = self._get_default_model_path()
        
        if not model_path.exists():
            try:
                download_model(model_name)
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
        
        tensor = np.array(img_resized, dtype=np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)
    
    def _upscale_mask(
        self,
        mask: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """resize mask to original frame dimensions."""
        from PIL import Image
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        
        resample = Image.BILINEAR if self.upscale_method == 'bilinear' else Image.NEAREST
        mask_resized = mask_pil.resize((target_w, target_h), resample)
        
        return np.array(mask_resized, dtype=np.float32) / 255.0
    
    def process(self, frame: np.ndarray) -> InferenceOutput:
        """
        segment person from background.
        
        args:
            frame: rgb image (h, w, 3) uint8
            
        returns:
            InferenceOutput with masks field containing (h, w) probability mask
        """
        if not self._initialized:
            if not self.initialize():
                return InferenceOutput(backend=InferenceBackend.NONE)
        
        if frame is None or frame.ndim < 3:
            return InferenceOutput(backend=self._backend)
        
        h, w = frame.shape[:2]
        
        # preprocess
        tensor = self._preprocess_frame(frame)
        
        # inference
        outputs = self._invoke(tensor)
        raw_mask = outputs[0].squeeze()
        
        # normalize if needed
        if raw_mask.min() < 0 or raw_mask.max() > 1:
            raw_mask = sigmoid(raw_mask)
        
        # upscale to original size
        mask = self._upscale_mask(raw_mask, h, w)
        
        return InferenceOutput(
            masks=mask,
            frame_width=w,
            frame_height=h,
            backend=self._backend,
        )
    
    def get_binary_mask(
        self,
        mask: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        convert probability mask to binary mask.
        
        args:
            mask: probability mask from process()
            threshold: cutoff (defaults to self.threshold)
            
        returns:
            binary mask as uint8 (0 or 255)
        """
        thresh = threshold if threshold is not None else self.threshold
        return ((mask > thresh) * 255).astype(np.uint8)
    
    def apply_background_blur(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        blur_size: int = 21,
    ) -> np.ndarray:
        """
        apply gaussian blur to background while keeping person sharp.
        
        args:
            frame: original rgb frame
            mask: probability mask from process()
            blur_size: gaussian kernel size (odd number)
            
        returns:
            composited frame with blurred background
        """
        try:
            import cv2
            blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        except ImportError:
            # fallback: simple box blur
            from PIL import Image, ImageFilter
            img = Image.fromarray(frame)
            blurred = np.array(img.filter(ImageFilter.GaussianBlur(blur_size // 2)))
        
        mask_3d = mask[..., np.newaxis]
        composite = (frame * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)
        
        return composite


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
