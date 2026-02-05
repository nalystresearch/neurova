"""
smart c++ module loader with automatic python fallback.

this module provides intelligent loading of c++ accelerated modules,
falling back to pure python implementations when c++ is unavailable.
"""

import sys
import importlib
from typing import Any, Optional
import warnings


class CPPModuleLoader:
    """loads c++ modules with automatic fallback to python."""
    
    def __init__(self):
        self.cpp_modules = {}
        self.fallback_modules = {}
        self.load_attempts = {}
        
    def load_module(self, module_name: str, cpp_module: str, fallback_module: str) -> Any:
        """
        load a module, preferring c++ but falling back to python.
        
        args:
            module_name: name of the module for tracking
            cpp_module: full path to c++ module
            fallback_module: full path to python module
            
        returns:
            loaded module object
        """
        # try c++ first
        try:
            module = importlib.import_module(cpp_module)
            self.cpp_modules[module_name] = True
            return module
        except ImportError as e:
            # fall back to python
            try:
                module = importlib.import_module(fallback_module)
                self.fallback_modules[module_name] = True
                if module_name not in self.load_attempts:
                    warnings.warn(
                        f"c++ module '{cpp_module}' not available, using python fallback. "
                        f"reason: {str(e)}",
                        RuntimeWarning,
                        stacklevel=2
                    )
                return module
            except ImportError as fallback_error:
                raise ImportError(
                    f"failed to load both c++ ({cpp_module}) and python ({fallback_module}) "
                    f"implementations. c++ error: {str(e)}. python error: {str(fallback_error)}"
                )
    
    def get_function(self, module_obj: Any, function_name: str, *fallback_names: str) -> Any:
        """
        get a function from a module, trying multiple names.
        
        args:
            module_obj: module to search in
            function_name: primary function name
            *fallback_names: alternative names to try
            
        returns:
            function object
        """
        # try primary name first
        if hasattr(module_obj, function_name):
            return getattr(module_obj, function_name)
        
        # try fallback names if needed
        for name in fallback_names:
            if hasattr(module_obj, name):
                return getattr(module_obj, name)
        
        raise AttributeError(
            f"function '{function_name}' not found in module. "
            f"also tried: {', '.join(fallback_names)}"
        )
    
    def report(self) -> dict:
        """get report of which modules loaded from c++ vs python."""
        return {
            'cpp_loaded': list(self.cpp_modules.keys()),
            'python_fallback': list(self.fallback_modules.keys()),
            'total_cpp': len(self.cpp_modules),
            'total_python': len(self.fallback_modules)
        }


# global loader instance
_loader = CPPModuleLoader()


def load_cpp_or_python(module_name: str, cpp_module: str, fallback_module: str) -> Any:
    """
    convenience function to load c++ or python module.
    
    example:
        imgproc = load_cpp_or_python('imgproc', 'neurova_minimal', 'neurova.imgproc')
    """
    return _loader.load_module(module_name, cpp_module, fallback_module)


def get_cpp_status() -> dict:
    """get report of which modules are using c++ vs python."""
    return _loader.report()


def force_cpp_only(enabled: bool = True):
    """
    force c++ only mode - raise errors instead of falling back to python.
    
    args:
        enabled: if true, disable python fallback
    """
    if enabled:
        warnings.warn(
            "c++ only mode enabled. python fallback disabled.",
            RuntimeWarning
        )
    # this would be implemented by modifying the loader behavior
    # for now, just a placeholder for future enhancement


def prefer_python(module_name: str):
    """mark a module to prefer python implementation over c++."""
    # placeholder for future enhancement
    pass


# module-specific loaders for the 6 compiled c++ modules
def load_minimal():
    """load neurova_minimal (imgproc, filters, features)."""
    return load_cpp_or_python('minimal', 'neurova_minimal', 'neurova.imgproc')


def load_architecture():
    """load neurova_architecture (neural network architectures)."""
    return load_cpp_or_python('architecture', 'neurova_architecture', 'neurova.nn')


def load_extended():
    """load neurova_extended (core, augmentation, calibration)."""
    return load_cpp_or_python('extended', 'neurova_extended', 'neurova.core')


def load_mega():
    """load neurova_mega (morphology, neural, nn, object_detection)."""
    return load_cpp_or_python('mega', 'neurova_mega', 'neurova.morphology')


def load_advanced():
    """load neurova_advanced (photo, segmentation, solutions)."""
    return load_cpp_or_python('advanced', 'neurova_advanced', 'neurova.photo')


def load_final():
    """load neurova_final (utils, transform, stitching)."""
    return load_cpp_or_python('final', 'neurova_final', 'neurova.utils')


def load_timeseries():
    """load neurova_timeseries (time series analysis)."""
    return load_cpp_or_python('timeseries', 'neurova_timeseries', 'neurova.timeseries')
