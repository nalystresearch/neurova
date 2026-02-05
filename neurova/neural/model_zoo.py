# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Model loading and weight management for Neurova."""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
import json
from neurova.neural.module import Module, Parameter
from neurova.core.errors import ValidationError


def save_weights(model: Module, filepath: Union[str, Path]) -> None:
    """Save model weights to NumPy format.
    
    Args:
        model: Module to save
        filepath: Output file path (.npz)
        
    Examples:
        model = MyModel()
        save_weights(model, "model_weights.npz")
    """
    filepath = Path(filepath)
    
    # collect all parameters
    params = model.parameters()
    
    # create dictionary of parameter arrays
    param_dict = {}
    for i, param in enumerate(params):
        param_dict[f"param_{i}"] = param.data
    
    # save to npz
    np.savez(str(filepath), **param_dict)


def load_weights(model: Module, filepath: Union[str, Path]) -> None:
    """Load model weights from NumPy format.
    
    Args:
        model: Module to load weights into
        filepath: Input file path (.npz)
        
    Examples:
        model = MyModel()
        load_weights(model, "model_weights.npz")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise ValidationError("filepath", str(filepath), "existing file")
    
    # load weights
    data = np.load(str(filepath))
    
    # get model parameters
    params = model.parameters()
    
    # load each parameter
    for i, param in enumerate(params):
        key = f"param_{i}"
        if key in data:
            param.data[:] = data[key]
        else:
            raise ValidationError(
                "weights file",
                f"missing parameter {key}",
                f"parameter {key} present"
            )


class ModelZoo:
    """Simple model zoo for pretrained weights.
    
    This class provides a registry for pretrained model weights stored as NumPy arrays.
    It does NOT include actual pretrained models (those would be too large).
    
    Instead, it provides infrastructure for:
    - Saving trained models
    - Loading saved models
    - Organizing model weights by name/version
    
    Examples:
        # train and save a model
        model = MyModel()
        # ... train model ...
        zoo = ModelZoo()
        zoo.save("my_model_v1", model)
        
        # load model later
        new_model = MyModel()
        zoo.load("my_model_v1", new_model)
    """
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize model zoo.
        
        Args:
            cache_dir: Directory to store model weights (default: ~/.neurova/models)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".neurova" / "models"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # load registry
        self.registry_path = self.cache_dir / "registry.json"
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry: Dict[str, Dict[str, Any]] = {}
    
    def save(self, name: str, model: Module, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model to zoo.
        
        Args:
            name: Model name/identifier
            model: Model to save
            metadata: Optional metadata (description, accuracy, etc.)
        """
        # save weights
        weights_path = self.cache_dir / f"{name}.npz"
        save_weights(model, weights_path)
        
        # update registry
        self.registry[name] = {
            "weights_path": str(weights_path),
            "metadata": metadata or {},
        }
        
        # save registry
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def load(self, name: str, model: Module) -> None:
        """Load model from zoo.
        
        Args:
            name: Model name/identifier
            model: Model to load weights into
        """
        if name not in self.registry:
            raise ValidationError("name", name, "registered model name")
        
        weights_path = Path(self.registry[name]["weights_path"])
        load_weights(model, weights_path)
    
    def list_models(self) -> list[str]:
        """List available models in zoo.
        
        Returns:
            List of model names
        """
        return list(self.registry.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get model metadata.
        
        Args:
            name: Model name/identifier
            
        Returns:
            Metadata dictionary
        """
        if name not in self.registry:
            raise ValidationError("name", name, "registered model name")
        
        return self.registry[name]["metadata"]


def export_to_onnx_compatible(model: Module, input_shape: tuple) -> Dict[str, np.ndarray]:
    """Export model to ONNX-compatible format (simplified).
    
    This is a placeholder for future ONNX export functionality.
    Currently just returns parameter dictionary.
    
    Args:
        model: Module to export
        input_shape: Example input shape for tracing
        
    Returns:
        Dictionary of parameter names to arrays
    """
    params = model.parameters()
    param_dict = {}
    for i, param in enumerate(params):
        param_dict[f"param_{i}"] = param.data
    
    return param_dict


__all__ = [
    "save_weights",
    "load_weights",
    "ModelZoo",
    "export_to_onnx_compatible",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.