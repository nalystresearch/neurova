# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Tkinter-based display backend for neurova.highgui.

This provides a pure-Python implementation for displaying images
using Tkinter and PIL, without requiring Neurova.
"""

from __future__ import annotations

import threading
import time
import queue
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field

import numpy as np

try:
    import tkinter as tk
    from PIL import Image, ImageTk
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

# Window state for tracking
@dataclass
class _TkWindowState:
    """State for a Tk window."""
    name: str
    root: Optional[Any] = None
    label: Optional[Any] = None
    photo: Optional[Any] = None
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


class TkBackend:
    """Tkinter-based backend for displaying images."""
    
    def __init__(self):
        self._windows: Dict[str, _TkWindowState] = {}
        self._last_key: int = -1
        self._key_lock = threading.Lock()
        self._update_queue: Dict[str, queue.Queue] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._root: Optional[tk.Tk] = None
        self._root_lock = threading.Lock()
    
    def _get_root(self) -> tk.Tk:
        """Get or create the main Tk root window."""
        with self._root_lock:
            if self._root is None:
                self._root = tk.Tk()
                self._root.withdraw()  # Hide the root window
            return self._root
    
    def namedWindow(self, winname: str, flags: int = 0) -> None:
        """Create a named window."""
        if winname not in self._windows:
            self._windows[winname] = _TkWindowState(name=winname, flags=flags)
            self._update_queue[winname] = queue.Queue()
    
    def imshow(self, winname: str, mat: np.ndarray) -> None:
        """Display an image in the named window."""
        if not _TK_AVAILABLE:
            return
        
        if winname not in self._windows:
            self.namedWindow(winname)
        
        state = self._windows[winname]
        
        # Ensure image is RGB (Tk expects RGB, but our images are BGR)
        if len(mat.shape) == 3 and mat.shape[2] == 3:
            # Convert BGR to RGB
            mat_rgb = mat[:, :, ::-1].copy()
        elif len(mat.shape) == 3 and mat.shape[2] == 4:
            # BGRA to RGBA
            mat_rgb = mat[:, :, [2, 1, 0, 3]].copy()
        elif len(mat.shape) == 2:
            # Grayscale - convert to RGB
            mat_rgb = np.stack([mat, mat, mat], axis=2)
        else:
            mat_rgb = mat.copy()
        
        state.image = mat_rgb
        
        # Create window if not exists
        if state.root is None:
            state.root = tk.Toplevel()
            state.root.title(winname)
            state.root.protocol("WM_DELETE_WINDOW", lambda: self._on_window_close(winname))
            state.label = tk.Label(state.root)
            state.label.pack()
            
            # Bind key events
            state.root.bind("<Key>", lambda e: self._on_key_press(e))
        
        # Update the image
        try:
            pil_image = Image.fromarray(mat_rgb)
            state.photo = ImageTk.PhotoImage(image=pil_image)
            state.label.configure(image=state.photo)
            state.root.update_idletasks()
            state.root.update()
        except Exception:
            pass
    
    def _on_window_close(self, winname: str) -> None:
        """Handle window close event."""
        if winname in self._windows:
            state = self._windows[winname]
            state.running = False
            if state.root:
                try:
                    state.root.destroy()
                except Exception:
                    pass
                state.root = None
    
    def _on_key_press(self, event) -> None:
        """Handle key press event."""
        with self._key_lock:
            self._last_key = ord(event.char) if event.char else event.keycode
    
    def waitKey(self, delay: int = 0) -> int:
        """Wait for a key press.
        
        Args:
            delay: Delay in milliseconds (0 = wait forever)
        
        Returns:
            Key code or -1 if no key pressed
        """
        if not _TK_AVAILABLE:
            if delay > 0:
                time.sleep(delay / 1000.0)
            return -1
        
        # Reset key state
        with self._key_lock:
            self._last_key = -1
        
        start_time = time.time()
        timeout = delay / 1000.0 if delay > 0 else float('inf')
        
        while True:
            # Update all windows
            for winname, state in self._windows.items():
                if state.root and state.running:
                    try:
                        state.root.update_idletasks()
                        state.root.update()
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
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
    
    def destroyWindow(self, winname: str) -> None:
        """Destroy a window."""
        if winname in self._windows:
            state = self._windows[winname]
            state.running = False
            if state.root:
                try:
                    state.root.destroy()
                except Exception:
                    pass
            del self._windows[winname]
        
        if winname in self._update_queue:
            del self._update_queue[winname]
    
    def destroyAllWindows(self) -> None:
        """Destroy all windows."""
        for winname in list(self._windows.keys()):
            self.destroyWindow(winname)
        
        with self._root_lock:
            if self._root:
                try:
                    self._root.destroy()
                except Exception:
                    pass
                self._root = None
    
    def moveWindow(self, winname: str, x: int, y: int) -> None:
        """Move a window to a new position."""
        if winname in self._windows:
            state = self._windows[winname]
            state.x = x
            state.y = y
            if state.root:
                try:
                    state.root.geometry(f"+{x}+{y}")
                except Exception:
                    pass
    
    def resizeWindow(self, winname: str, width: int, height: int) -> None:
        """Resize a window."""
        if winname in self._windows:
            state = self._windows[winname]
            state.width = width
            state.height = height
            if state.root:
                try:
                    state.root.geometry(f"{width}x{height}")
                except Exception:
                    pass


# Global backend instance
_backend_instance: Optional[TkBackend] = None


def _get_backend() -> TkBackend:
    """Get or create the backend instance."""
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = TkBackend()
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
