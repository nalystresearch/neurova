# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
neurova.highgui - High-level GUI functions for image display and interaction

This module provides GUI functions for displaying images,
creating windows, and handling user input.
"""

from __future__ import annotations

import sys
import time
import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np

# Window flags
WINDOW_NORMAL = 0x00000000
WINDOW_AUTOSIZE = 0x00000001
WINDOW_OPENGL = 0x00001000
WINDOW_FULLSCREEN = 1
WINDOW_FREERATIO = 0x00000100
WINDOW_KEEPRATIO = 0x00000000
WINDOW_GUI_EXPANDED = 0x00000000
WINDOW_GUI_NORMAL = 0x00000010

# Window property constants (for getWindowProperty/setWindowProperty)
WND_PROP_FULLSCREEN = 0
WND_PROP_AUTOSIZE = 1
WND_PROP_ASPECT_RATIO = 2
WND_PROP_OPENGL = 3
WND_PROP_VISIBLE = 4
WND_PROP_TOPMOST = 5
WND_PROP_VSYNC = 6

# Mouse event types
EVENT_MOUSEMOVE = 0
EVENT_LBUTTONDOWN = 1
EVENT_RBUTTONDOWN = 2
EVENT_MBUTTONDOWN = 3
EVENT_LBUTTONUP = 4
EVENT_RBUTTONUP = 5
EVENT_MBUTTONUP = 6
EVENT_LBUTTONDBLCLK = 7
EVENT_RBUTTONDBLCLK = 8
EVENT_MBUTTONDBLCLK = 9
EVENT_MOUSEWHEEL = 10
EVENT_MOUSEHWHEEL = 11

# Mouse event flags
EVENT_FLAG_LBUTTON = 1
EVENT_FLAG_RBUTTON = 2
EVENT_FLAG_MBUTTON = 4
EVENT_FLAG_CTRLKEY = 8
EVENT_FLAG_SHIFTKEY = 16
EVENT_FLAG_ALTKEY = 32

# Qt button types
QT_PUSH_BUTTON = 0x00000000
QT_CHECKBOX = 0x00000001
QT_RADIOBOX = 0x00000002
QT_NEW_BUTTONBAR = 0x00000004


@dataclass
class Trackbar:
    """Trackbar (slider) state."""
    name: str
    value: int
    max_value: int
    callback: Optional[Callable[[int], None]] = None


@dataclass
class WindowState:
    """Internal window state."""
    name: str
    image: Optional[np.ndarray] = None
    flags: int = WINDOW_AUTOSIZE
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 480
    trackbars: Dict[str, Trackbar] = field(default_factory=dict)
    mouse_callback: Optional[Callable[[int, int, int, int, Any], None]] = None
    mouse_userdata: Any = None
    visible: bool = True


# Global window registry
_windows: Dict[str, WindowState] = {}
_last_key: int = -1
_key_lock = threading.Lock()

# Backend selection
_backend: Optional[str] = None
_backend_module: Any = None


def _get_backend():
    """Get or initialize the display backend."""
    global _backend, _backend_module
    
    if _backend_module is not None:
        return _backend_module
    
    # Try different backends in order of preference
    # Prefer matplotlib backend as it's more widely available
    backends_to_try = ["matplotlib", "tk"]
    
    for backend_name in backends_to_try:
        try:
            if backend_name == "matplotlib":
                from neurova.highgui import _matplotlib_backend as mod
                _backend = "matplotlib"
                _backend_module = mod
                return mod
            elif backend_name == "tk":
                from neurova.highgui import _tk_backend as mod
                _backend = "tk"
                _backend_module = mod
                return mod
        except ImportError:
            continue
    
    # Fallback to headless mode
    _backend = "headless"
    _backend_module = _HeadlessBackend()
    warnings.warn("No GUI backend available, running in headless mode")
    return _backend_module


class _HeadlessBackend:
    """Headless backend for environments without display."""
    
    def namedWindow(self, winname: str, flags: int = WINDOW_AUTOSIZE) -> None:
        if winname not in _windows:
            _windows[winname] = WindowState(name=winname, flags=flags)
    
    def imshow(self, winname: str, mat: np.ndarray) -> None:
        if winname not in _windows:
            self.namedWindow(winname)
        _windows[winname].image = mat.copy()
    
    def waitKey(self, delay: int = 0) -> int:
        if delay > 0:
            time.sleep(delay / 1000.0)
        return -1
    
    def destroyWindow(self, winname: str) -> None:
        if winname in _windows:
            del _windows[winname]
    
    def destroyAllWindows(self) -> None:
        _windows.clear()
    
    def moveWindow(self, winname: str, x: int, y: int) -> None:
        if winname in _windows:
            _windows[winname].x = x
            _windows[winname].y = y
    
    def resizeWindow(self, winname: str, width: int, height: int) -> None:
        if winname in _windows:
            _windows[winname].width = width
            _windows[winname].height = height


class _NativeBackend:
    """Backend using native display module."""
    
    def __init__(self):
        from neurova.video.display_native import NativeDisplay
        self._NativeDisplay = NativeDisplay
        self._displays: Dict[str, Any] = {}
        self._last_key: int = -1
    
    def namedWindow(self, winname: str, flags: int = WINDOW_AUTOSIZE) -> None:
        if winname not in _windows:
            _windows[winname] = WindowState(name=winname, flags=flags)
    
    def imshow(self, winname: str, mat: np.ndarray) -> None:
        if winname not in _windows:
            self.namedWindow(winname)
        
        state = _windows[winname]
        state.image = mat.copy()
        
        # Use native display
        try:
            if winname not in self._displays:
                display = self._NativeDisplay()
                display.open()
                self._displays[winname] = display
            
            self._displays[winname].show(mat)
        except Exception as e:
            # Fallback to just storing the image
            pass
    
    def waitKey(self, delay: int = 0) -> int:
        # Process events for all displays
        if delay > 0:
            time.sleep(delay / 1000.0)
        return -1
    
    def destroyWindow(self, winname: str) -> None:
        if winname in self._displays:
            try:
                self._displays[winname].close()
            except Exception:
                pass
            del self._displays[winname]
        if winname in _windows:
            del _windows[winname]
    
    def destroyAllWindows(self) -> None:
        for winname in list(self._displays.keys()):
            self.destroyWindow(winname)
        _windows.clear()
    
    def moveWindow(self, winname: str, x: int, y: int) -> None:
        if winname in _windows:
            _windows[winname].x = x
            _windows[winname].y = y
    
    def resizeWindow(self, winname: str, width: int, height: int) -> None:
        if winname in _windows:
            _windows[winname].width = width
            _windows[winname].height = height


# Public API Functions

def namedWindow(winname: str, flags: int = WINDOW_AUTOSIZE) -> None:
    """Create a window that can be used as a placeholder for images and trackbars.
    
    Args:
        winname: Name of the window (used as identifier)
        flags: Window flags (WINDOW_NORMAL, WINDOW_AUTOSIZE, etc.)
    """
    backend = _get_backend()
    backend.namedWindow(winname, flags)


def imshow(winname: str, mat: np.ndarray) -> None:
    """Display an image in the specified window.
    
    Args:
        winname: Name of the window
        mat: Image to be shown (numpy array)
    """
    backend = _get_backend()
    
    # Convert to uint8 if needed
    if mat.dtype != np.uint8:
        if mat.dtype in (np.float32, np.float64):
            mat = (np.clip(mat, 0, 1) * 255).astype(np.uint8)
        else:
            mat = mat.astype(np.uint8)
    
    backend.imshow(winname, mat)


def waitKey(delay: int = 0) -> int:
    """Wait for a pressed key.
    
    Args:
        delay: Delay in milliseconds (0 = wait forever)
    
    Returns:
        Code of the pressed key or -1 if no key was pressed
    """
    backend = _get_backend()
    return backend.waitKey(delay)


def waitKeyEx(delay: int = 0) -> int:
    """Wait for a pressed key (extended version with full keycode).
    
    Args:
        delay: Delay in milliseconds (0 = wait forever)
    
    Returns:
        Full keycode of the pressed key or -1 if no key was pressed
    """
    return waitKey(delay)


def destroyWindow(winname: str) -> None:
    """Destroy a window.
    
    Args:
        winname: Name of the window to destroy
    """
    backend = _get_backend()
    backend.destroyWindow(winname)


def destroyAllWindows() -> None:
    """Destroy all HighGUI windows."""
    backend = _get_backend()
    backend.destroyAllWindows()


def moveWindow(winname: str, x: int, y: int) -> None:
    """Move window to the specified position.
    
    Args:
        winname: Name of the window
        x: New x-coordinate of the window
        y: New y-coordinate of the window
    """
    backend = _get_backend()
    backend.moveWindow(winname, x, y)


def resizeWindow(winname: str, width: int, height: int) -> None:
    """Resize a window.
    
    Args:
        winname: Name of the window
        width: New width
        height: New height
    """
    backend = _get_backend()
    backend.resizeWindow(winname, width, height)


def setWindowTitle(winname: str, title: str) -> None:
    """Update window title.
    
    Args:
        winname: Name of the window
        title: New title
    """
    if winname in _windows:
        _windows[winname].name = title


def getWindowProperty(winname: str, prop_id: int) -> float:
    """Get window property.
    
    Args:
        winname: Name of the window
        prop_id: Property identifier
    
    Returns:
        Property value
    """
    if winname in _windows:
        state = _windows[winname]
        if prop_id == 0:  # WND_PROP_FULLSCREEN
            return 0.0
        elif prop_id == 1:  # WND_PROP_AUTOSIZE
            return float(state.flags & WINDOW_AUTOSIZE)
        elif prop_id == 2:  # WND_PROP_ASPECT_RATIO
            return float(state.width) / max(1, state.height)
        elif prop_id == 3:  # WND_PROP_OPENGL
            return float(state.flags & WINDOW_OPENGL)
        elif prop_id == 4:  # WND_PROP_VISIBLE
            return 1.0 if state.visible else 0.0
    return -1.0


def setWindowProperty(winname: str, prop_id: int, prop_value: float) -> None:
    """Set window property.
    
    Args:
        winname: Name of the window
        prop_id: Property identifier
        prop_value: New property value
    """
    pass  # Most properties cannot be changed after creation


def createTrackbar(
    trackbarname: str,
    winname: str,
    value: int,
    count: int,
    onChange: Optional[Callable[[int], None]] = None
) -> int:
    """Create a trackbar and attach it to the specified window.
    
    Args:
        trackbarname: Name of the trackbar
        winname: Name of the window to attach to
        value: Initial value of the trackbar
        count: Maximum value of the trackbar
        onChange: Callback function called when the trackbar position changes
    
    Returns:
        Initial position of the trackbar
    """
    if winname not in _windows:
        namedWindow(winname)
    
    trackbar = Trackbar(
        name=trackbarname,
        value=value,
        max_value=count,
        callback=onChange
    )
    _windows[winname].trackbars[trackbarname] = trackbar
    
    return value


def getTrackbarPos(trackbarname: str, winname: str) -> int:
    """Get current trackbar position.
    
    Args:
        trackbarname: Name of the trackbar
        winname: Name of the window
    
    Returns:
        Current position of the trackbar
    """
    if winname in _windows and trackbarname in _windows[winname].trackbars:
        return _windows[winname].trackbars[trackbarname].value
    return 0


def setTrackbarPos(trackbarname: str, winname: str, pos: int) -> None:
    """Set trackbar position.
    
    Args:
        trackbarname: Name of the trackbar
        winname: Name of the window
        pos: New position
    """
    if winname in _windows and trackbarname in _windows[winname].trackbars:
        trackbar = _windows[winname].trackbars[trackbarname]
        trackbar.value = max(0, min(pos, trackbar.max_value))
        if trackbar.callback is not None:
            trackbar.callback(trackbar.value)


def setTrackbarMax(trackbarname: str, winname: str, maxval: int) -> None:
    """Set maximum value of trackbar.
    
    Args:
        trackbarname: Name of the trackbar
        winname: Name of the window
        maxval: New maximum value
    """
    if winname in _windows and trackbarname in _windows[winname].trackbars:
        _windows[winname].trackbars[trackbarname].max_value = maxval


def setTrackbarMin(trackbarname: str, winname: str, minval: int) -> None:
    """Set minimum value of trackbar.
    
    Args:
        trackbarname: Name of the trackbar
        winname: Name of the window
        minval: New minimum value (usually 0)
    """
    pass  # Neurova trackbars always start at 0


def setMouseCallback(
    winname: str,
    onMouse: Callable[[int, int, int, int, Any], None],
    param: Any = None
) -> None:
    """Set mouse handler for the specified window.
    
    Args:
        winname: Name of the window
        onMouse: Callback function(event, x, y, flags, param)
        param: User data passed to the callback
    """
    if winname not in _windows:
        namedWindow(winname)
    
    _windows[winname].mouse_callback = onMouse
    _windows[winname].mouse_userdata = param


def getMouseWheelDelta(flags: int) -> int:
    """Get mouse wheel motion delta.
    
    Args:
        flags: Mouse event flags
    
    Returns:
        Wheel delta value
    """
    return (flags >> 16) & 0xFFFF


def selectROI(
    windowName: str,
    img: np.ndarray,
    showCrosshair: bool = True,
    fromCenter: bool = False
) -> Tuple[int, int, int, int]:
    """Select a Region of Interest (ROI) using a mouse.
    
    Args:
        windowName: Name of the window
        img: Image to select ROI from
        showCrosshair: Draw crosshair in the selection rectangle
        fromCenter: If true, selection starts from the center
    
    Returns:
        Tuple (x, y, width, height) of the selected rectangle
    """
    # Simplified implementation - just returns the full image
    warnings.warn("selectROI: interactive selection not available, returning full image")
    return (0, 0, img.shape[1], img.shape[0])


def selectROIs(
    windowName: str,
    img: np.ndarray,
    showCrosshair: bool = True,
    fromCenter: bool = False
) -> List[Tuple[int, int, int, int]]:
    """Select multiple Regions of Interest.
    
    Args:
        windowName: Name of the window
        img: Image to select ROIs from
        showCrosshair: Draw crosshair
        fromCenter: If true, selection starts from the center
    
    Returns:
        List of ROI tuples
    """
    roi = selectROI(windowName, img, showCrosshair, fromCenter)
    return [roi]


# Exports

__all__ = [
    # Window flags
    "WINDOW_NORMAL",
    "WINDOW_AUTOSIZE",
    "WINDOW_OPENGL",
    "WINDOW_FULLSCREEN",
    "WINDOW_FREERATIO",
    "WINDOW_KEEPRATIO",
    "WINDOW_GUI_EXPANDED",
    "WINDOW_GUI_NORMAL",
    # Window property constants
    "WND_PROP_FULLSCREEN",
    "WND_PROP_AUTOSIZE",
    "WND_PROP_ASPECT_RATIO",
    "WND_PROP_OPENGL",
    "WND_PROP_VISIBLE",
    "WND_PROP_TOPMOST",
    "WND_PROP_VSYNC",
    # Mouse events
    "EVENT_MOUSEMOVE",
    "EVENT_LBUTTONDOWN",
    "EVENT_RBUTTONDOWN",
    "EVENT_MBUTTONDOWN",
    "EVENT_LBUTTONUP",
    "EVENT_RBUTTONUP",
    "EVENT_MBUTTONUP",
    "EVENT_LBUTTONDBLCLK",
    "EVENT_RBUTTONDBLCLK",
    "EVENT_MBUTTONDBLCLK",
    "EVENT_MOUSEWHEEL",
    "EVENT_MOUSEHWHEEL",
    # Mouse flags
    "EVENT_FLAG_LBUTTON",
    "EVENT_FLAG_RBUTTON",
    "EVENT_FLAG_MBUTTON",
    "EVENT_FLAG_CTRLKEY",
    "EVENT_FLAG_SHIFTKEY",
    "EVENT_FLAG_ALTKEY",
    # Functions
    "namedWindow",
    "imshow",
    "waitKey",
    "waitKeyEx",
    "destroyWindow",
    "destroyAllWindows",
    "moveWindow",
    "resizeWindow",
    "setWindowTitle",
    "getWindowProperty",
    "setWindowProperty",
    "createTrackbar",
    "getTrackbarPos",
    "setTrackbarPos",
    "setTrackbarMax",
    "setTrackbarMin",
    "setMouseCallback",
    "getMouseWheelDelta",
    "selectROI",
    "selectROIs",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.