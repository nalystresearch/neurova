# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Matplotlib-based display backend for neurova.highgui.

This provides a pure-Python implementation for displaying images
using Matplotlib, without requiring Neurova or Tkinter.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field

import numpy as np

try:
    import matplotlib
    matplotlib.use('MacOSX')  # Use native macOS backend
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


@dataclass
class _MplWindowState:
    """State for a matplotlib figure window."""
    name: str
    fig: Optional[Any] = None
    ax: Optional[Any] = None
    im: Optional[Any] = None
    image: Optional[np.ndarray] = None
    running: bool = True
    flags: int = 0
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 480
    trackbars: Dict[str, Any] = field(default_factory=dict)
    mouse_callback: Optional[Callable] = None
    mouse_userdata: Any = None


class MatplotlibBackend:
    """Matplotlib-based backend for displaying images."""
    
    def __init__(self):
        self._windows: Dict[str, _MplWindowState] = {}
        self._last_key: int = -1
        self._key_lock = threading.Lock()
        
        if _MPL_AVAILABLE:
            plt.ion()  # Enable interactive mode
    
    def namedWindow(self, winname: str, flags: int = 0) -> None:
        """Create a named window."""
        if winname not in self._windows:
            self._windows[winname] = _MplWindowState(name=winname, flags=flags)
    
    def imshow(self, winname: str, mat: np.ndarray) -> None:
        """Display an image in the named window."""
        if not _MPL_AVAILABLE:
            return
        
        if winname not in self._windows:
            self.namedWindow(winname)
        
        state = self._windows[winname]
        
        # Neurova uses RGB format directly (not BGR like Neurova)
        # So we pass the image directly to matplotlib without conversion
        if len(mat.shape) == 3 and mat.shape[2] == 3:
            mat_rgb = mat.copy()  # Already RGB
        elif len(mat.shape) == 3 and mat.shape[2] == 4:
            mat_rgb = mat[:, :, :3].copy()  # RGBA -> RGB (drop alpha)
        elif len(mat.shape) == 2:
            mat_rgb = mat
        else:
            mat_rgb = mat.copy()
        
        state.image = mat_rgb
        
        # Create figure if not exists
        if state.fig is None:
            state.fig, state.ax = plt.subplots(figsize=(mat.shape[1]/100, mat.shape[0]/100))
            state.fig.canvas.manager.set_window_title(winname)
            state.ax.axis('off')
            state.im = state.ax.imshow(mat_rgb)
            
            # Connect key press event
            state.fig.canvas.mpl_connect('key_press_event', 
                lambda e: self._on_key_press(e))
            
            plt.tight_layout(pad=0)
            plt.show(block=False)
        else:
            # Update existing image
            state.im.set_data(mat_rgb)
            state.fig.canvas.draw_idle()
            state.fig.canvas.flush_events()
    
    def _on_key_press(self, event) -> None:
        """Handle key press event."""
        with self._key_lock:
            if event.key and len(event.key) == 1:
                self._last_key = ord(event.key)
            elif event.key == 'escape':
                self._last_key = 27
            elif event.key == 'enter':
                self._last_key = 13
    
    def waitKey(self, delay: int = 0) -> int:
        """Wait for a key press.
        
        Args:
            delay: Delay in milliseconds (0 = wait forever)
        
        Returns:
            Key code or -1 if no key pressed
        """
        if not _MPL_AVAILABLE:
            if delay > 0:
                time.sleep(delay / 1000.0)
            return -1
        
        # Reset key state
        with self._key_lock:
            self._last_key = -1
        
        start_time = time.time()
        timeout = delay / 1000.0 if delay > 0 else float('inf')
        
        while True:
            # Process matplotlib events
            try:
                plt.pause(0.001)
            except Exception:
                pass
            
            # Check for key press
            with self._key_lock:
                if self._last_key != -1:
                    key = self._last_key
                    self._last_key = -1
                    return key
            
            # Check timeout
            if time.time() - start_time >= timeout:
                return -1
    
    def destroyWindow(self, winname: str) -> None:
        """Destroy a window."""
        if winname in self._windows:
            state = self._windows[winname]
            state.running = False
            if state.fig is not None:
                try:
                    plt.close(state.fig)
                except Exception:
                    pass
            del self._windows[winname]
    
    def destroyAllWindows(self) -> None:
        """Destroy all windows."""
        for winname in list(self._windows.keys()):
            self.destroyWindow(winname)
        
        try:
            plt.close('all')
        except Exception:
            pass
    
    def moveWindow(self, winname: str, x: int, y: int) -> None:
        """Move a window to a new position."""
        if winname in self._windows:
            state = self._windows[winname]
            state.x = x
            state.y = y
    
    def resizeWindow(self, winname: str, width: int, height: int) -> None:
        """Resize a window."""
        if winname in self._windows:
            state = self._windows[winname]
            state.width = width
            state.height = height


# Global backend instance
_backend_instance: Optional[MatplotlibBackend] = None


def _get_backend() -> MatplotlibBackend:
    """Get or create the backend instance."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = MatplotlibBackend()
    return _backend_instance


# Module-level convenience functions
def namedWindow(winname: str, flags: int = 0) -> None:
    """Create a named window."""
    _get_backend().namedWindow(winname, flags)


def imshow(winname: str, mat: np.ndarray) -> None:
    """Display an image in a window."""
    _get_backend().imshow(winname, mat)


def waitKey(delay: int = 0) -> int:
    """Wait for a key press."""
    return _get_backend().waitKey(delay)


def destroyWindow(winname: str) -> None:
    """Destroy a window."""
    _get_backend().destroyWindow(winname)


def destroyAllWindows() -> None:
    """Destroy all windows."""
    _get_backend().destroyAllWindows()


def moveWindow(winname: str, x: int, y: int) -> None:
    """Move a window."""
    _get_backend().moveWindow(winname, x, y)


def resizeWindow(winname: str, width: int, height: int) -> None:
    """Resize a window."""
    _get_backend().resizeWindow(winname, width, height)
