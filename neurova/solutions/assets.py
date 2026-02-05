# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
model manager for neurova solutions.

handles downloading, caching, and loading of tflite models.
models are stored in the neurova package directory for offline use.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# base url for model assets on google storage
_MODELS_BASE = "https://storage.googleapis.com/neurova-models"

# model urls for all solutions
MODEL_URLS: Dict[str, str] = {
    # face detection
    "blaze_face_short_range": (
        f"{_MODELS_BASE}/face_detector/blaze_face_short_range/float16/1/"
        "blaze_face_short_range.tflite"
    ),
    # face mesh (468 landmarks)
    "face_landmark": (
        f"{_MODELS_BASE}/face_landmarker/face_landmarker/float16/1/"
        "face_landmarker.task"
    ),
    "face_landmark_lite": (
        "https://storage.googleapis.com/model-assets/"
        "face_landmark.tflite"
    ),
    # hand detection and landmarks
    "palm_detector": (
        f"{_MODELS_BASE}/hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task"
    ),
    "palm_detector_lite": (
        "https://storage.googleapis.com/model-assets/"
        "palm_detection_lite.tflite"
    ),
    "hand_landmark": (
        "https://storage.googleapis.com/model-assets/"
        "hand_landmark_lite.tflite"
    ),
    # pose detection (33 landmarks)
    "pose_detector": (
        "https://storage.googleapis.com/model-assets/"
        "pose_detection.tflite"
    ),
    "pose_landmark_lite": (
        "https://storage.googleapis.com/model-assets/"
        "pose_landmark_lite.tflite"
    ),
    "pose_landmark_full": (
        "https://storage.googleapis.com/model-assets/"
        "pose_landmark_full.tflite"
    ),
    "pose_landmark_heavy": (
        "https://storage.googleapis.com/model-assets/"
        "pose_landmark_heavy.tflite"
    ),
    # selfie segmentation
    "selfie_segmentation": (
        "https://storage.googleapis.com/model-assets/"
        "selfie_segmentation.tflite"
    ),
    "selfie_segmentation_landscape": (
        "https://storage.googleapis.com/model-assets/"
        "selfie_segmentation_landscape.tflite"
    ),
    # hair segmentation
    "hair_segmentation": (
        "https://storage.googleapis.com/model-assets/"
        "hair_segmentation.tflite"
    ),
    # iris detection
    "iris_landmark": (
        "https://storage.googleapis.com/model-assets/"
        "iris_landmark.tflite"
    ),
}

# expected model sizes for validation (approximate bytes)
MODEL_SIZES: Dict[str, int] = {
    "blaze_face_short_range": 230000,
    "face_landmark_lite": 2800000,
    "palm_detector_lite": 2000000,
    "hand_landmark": 3500000,
    "pose_detector": 2500000,
    "pose_landmark_lite": 3500000,
    "pose_landmark_full": 6000000,
    "selfie_segmentation": 260000,
    "selfie_segmentation_landscape": 260000,
    "hair_segmentation": 300000,
    "iris_landmark": 2600000,
}


def get_models_dir() -> Path:
    """
    get the directory where models are stored.
    
    returns the neurova/solutions/weights directory.
    """
    return Path(__file__).resolve().parent / "weights"


def get_model_path(model_name: str) -> Path:
    """
    get the local path for a model.
    
    args:
        model_name: name of the model (key in MODEL_URLS)
        
    returns:
        path to the model file
    """
    models_dir = get_models_dir()
    return models_dir / f"{model_name}.tflite"


def download_model(
    model_name: str,
    force: bool = False,
    progress_callback: Optional[callable] = None,
) -> Path:
    """
    download a model if not already present.
    
    args:
        model_name: name of the model to download
        force: if true, re-download even if exists
        progress_callback: optional callback(downloaded, total)
        
    returns:
        path to the downloaded model
        
    raises:
        valueerror: if model name is unknown
        runtimeerror: if download fails
    """
    if model_name not in MODEL_URLS:
        raise ValueError(f"unknown model: {model_name}")
    
    model_path = get_model_path(model_name)
    
    # return if already exists and not forcing
    if model_path.exists() and not force:
        return model_path
    
    # create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # download
    url = MODEL_URLS[model_name]
    temp_path = model_path.with_suffix(".tmp")
    
    try:
        if progress_callback:
            _download_with_progress(url, temp_path, progress_callback)
        else:
            urllib.request.urlretrieve(url, str(temp_path))
        
        # move to final location
        shutil.move(str(temp_path), str(model_path))
        return model_path
        
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"failed to download {model_name}: {e}")


def _download_with_progress(
    url: str,
    target: Path,
    callback: callable,
) -> None:
    """download file with progress callback."""
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192
        
        with open(target, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                callback(downloaded, total)


def verify_model(model_name: str) -> bool:
    """
    verify that a model exists and has reasonable size.
    
    args:
        model_name: name of the model to verify
        
    returns:
        true if model exists and appears valid
    """
    model_path = get_model_path(model_name)
    
    if not model_path.exists():
        return False
    
    # check size is reasonable
    size = model_path.stat().st_size
    if size < 1000:
        return False
    
    # check expected size if known
    if model_name in MODEL_SIZES:
        expected = MODEL_SIZES[model_name]
        # allow 50% variance
        if size < expected * 0.5 or size > expected * 2:
            return False
    
    return True


def list_available_models() -> Dict[str, bool]:
    """
    list all known models and their availability.
    
    returns:
        dict mapping model names to whether they are downloaded
    """
    return {
        name: get_model_path(name).exists()
        for name in MODEL_URLS.keys()
    }


def download_all_models(
    progress_callback: Optional[callable] = None,
) -> Dict[str, bool]:
    """
    download all models.
    
    args:
        progress_callback: optional callback(model_name, downloaded, total)
        
    returns:
        dict mapping model names to download success
    """
    results = {}
    
    for name in MODEL_URLS.keys():
        try:
            download_model(name, progress_callback=progress_callback)
            results[name] = True
        except Exception:
            results[name] = False
    
    return results


class ModelManager:
    """
    manager for neurova solution models.
    
    provides methods to download, verify, and load tflite models.
    
    example:
        manager = ModelManager()
        
        # download a specific model
        path = manager.download("pose_landmark_lite")
        
        # check if model is available
        if manager.is_available("hand_landmark"):
            ...
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        initialize the model manager.
        
        args:
            models_dir: custom directory for models, uses default if none
        """
        self.models_dir = models_dir or get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        model_name: str,
        force: bool = False,
    ) -> Path:
        """
        download a model.
        
        args:
            model_name: name of the model
            force: re-download even if exists
            
        returns:
            path to the model file
        """
        return download_model(model_name, force=force)
    
    def is_available(self, model_name: str) -> bool:
        """check if a model is downloaded and valid."""
        return verify_model(model_name)
    
    def get_path(self, model_name: str) -> Path:
        """get the local path for a model."""
        return get_model_path(model_name)
    
    def list_models(self) -> Dict[str, bool]:
        """list all models and their availability."""
        return list_available_models()
    
    def ensure_model(self, model_name: str) -> Path:
        """
        ensure a model is available, downloading if needed.
        
        args:
            model_name: name of the model
            
        returns:
            path to the model file
        """
        if not self.is_available(model_name):
            return self.download(model_name)
        return self.get_path(model_name)


# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.
