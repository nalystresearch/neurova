# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
FFT Module.

Neurova implementation of Fast Fourier Transform operations
for signal processing and spectral analysis in deep learning.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union, List, Sequence


# 1D Fourier Transforms

def fft(x: np.ndarray, n: Optional[int] = None, dim: int = -1, norm: str = 'backward') -> np.ndarray:
    """
    Compute 1D Discrete Fourier Transform.
    
    Args:
        x: Input tensor
        n: Signal length (zero-padded or truncated if different from input)
        dim: Dimension over which to compute FFT
        norm: Normalization mode ('forward', 'backward', 'ortho')
        
    Returns:
        Complex tensor of FFT coefficients
    """
    result = np.fft.fft(x, n=n, axis=dim)
    return _apply_norm(result, n or x.shape[dim], norm, forward=True)


def ifft(x: np.ndarray, n: Optional[int] = None, dim: int = -1, norm: str = 'backward') -> np.ndarray:
    """
    Compute 1D Inverse Discrete Fourier Transform.
    
    Args:
        x: Input complex tensor
        n: Signal length
        dim: Dimension over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Complex tensor of inverse FFT
    """
    result = np.fft.ifft(x, n=n, axis=dim)
    return _apply_norm(result, n or x.shape[dim], norm, forward=False)


def rfft(x: np.ndarray, n: Optional[int] = None, dim: int = -1, norm: str = 'backward') -> np.ndarray:
    """
    Compute 1D FFT of real-valued input.
    
    Args:
        x: Real input tensor
        n: Signal length
        dim: Dimension over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Complex tensor with positive frequencies only
    """
    result = np.fft.rfft(x, n=n, axis=dim)
    n_actual = n or x.shape[dim]
    return _apply_norm(result, n_actual, norm, forward=True)


def irfft(x: np.ndarray, n: Optional[int] = None, dim: int = -1, norm: str = 'backward') -> np.ndarray:
    """
    Compute inverse FFT of real signal.
    
    Args:
        x: Complex input tensor (positive frequencies)
        n: Output signal length
        dim: Dimension over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Real-valued tensor
    """
    result = np.fft.irfft(x, n=n, axis=dim)
    n_actual = n or 2 * (x.shape[dim] - 1)
    return _apply_norm(result, n_actual, norm, forward=False)


def hfft(x: np.ndarray, n: Optional[int] = None, dim: int = -1, norm: str = 'backward') -> np.ndarray:
    """
    Compute FFT of Hermitian symmetric signal.
    
    Args:
        x: Input tensor with Hermitian symmetry
        n: Output signal length
        dim: Dimension over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Real-valued tensor
    """
    result = np.fft.hfft(x, n=n, axis=dim)
    n_actual = n or 2 * (x.shape[dim] - 1)
    return _apply_norm(result, n_actual, norm, forward=True)


def ihfft(x: np.ndarray, n: Optional[int] = None, dim: int = -1, norm: str = 'backward') -> np.ndarray:
    """
    Compute inverse FFT producing Hermitian symmetric output.
    
    Args:
        x: Real input tensor
        n: Signal length
        dim: Dimension over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Complex tensor with Hermitian symmetry
    """
    result = np.fft.ihfft(x, n=n, axis=dim)
    n_actual = n or x.shape[dim]
    return _apply_norm(result, n_actual, norm, forward=False)


# 2D Fourier Transforms

def fft2(x: np.ndarray, s: Optional[Tuple[int, int]] = None, 
         dim: Tuple[int, int] = (-2, -1), norm: str = 'backward') -> np.ndarray:
    """
    Compute 2D Discrete Fourier Transform.
    
    Args:
        x: Input tensor
        s: Shape of output (zero-padded or truncated)
        dim: Dimensions over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Complex tensor of 2D FFT coefficients
    """
    result = np.fft.fft2(x, s=s, axes=dim)
    n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=True)


def ifft2(x: np.ndarray, s: Optional[Tuple[int, int]] = None,
          dim: Tuple[int, int] = (-2, -1), norm: str = 'backward') -> np.ndarray:
    """
    Compute 2D Inverse Discrete Fourier Transform.
    
    Args:
        x: Input complex tensor
        s: Shape of output
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Complex tensor of inverse 2D FFT
    """
    result = np.fft.ifft2(x, s=s, axes=dim)
    n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=False)


def rfft2(x: np.ndarray, s: Optional[Tuple[int, int]] = None,
          dim: Tuple[int, int] = (-2, -1), norm: str = 'backward') -> np.ndarray:
    """
    Compute 2D FFT of real-valued input.
    
    Args:
        x: Real input tensor
        s: Shape of output
        dim: Dimensions over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Complex tensor with positive frequencies
    """
    result = np.fft.rfft2(x, s=s, axes=dim)
    n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=True)


def irfft2(x: np.ndarray, s: Optional[Tuple[int, int]] = None,
           dim: Tuple[int, int] = (-2, -1), norm: str = 'backward') -> np.ndarray:
    """
    Compute inverse 2D FFT of real signal.
    
    Args:
        x: Complex input tensor
        s: Output shape
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Real-valued tensor
    """
    result = np.fft.irfft2(x, s=s, axes=dim)
    n_total = np.prod(s if s else [2 * (x.shape[dim[-1]] - 1), x.shape[dim[-2]]])
    return _apply_norm(result, n_total, norm, forward=False)


def hfft2(x: np.ndarray, s: Optional[Tuple[int, int]] = None,
          dim: Tuple[int, int] = (-2, -1), norm: str = 'backward') -> np.ndarray:
    """
    Compute 2D FFT of Hermitian symmetric signal.
    
    Args:
        x: Input tensor with Hermitian symmetry
        s: Output shape
        dim: Dimensions over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Real-valued tensor
    """
    return np.real(fft2(x, s=s, dim=dim, norm=norm))


def ihfft2(x: np.ndarray, s: Optional[Tuple[int, int]] = None,
           dim: Tuple[int, int] = (-2, -1), norm: str = 'backward') -> np.ndarray:
    """
    Compute inverse 2D FFT producing Hermitian symmetric output.
    """
    return ifft2(x.astype(complex), s=s, dim=dim, norm=norm)


# N-D Fourier Transforms

def fftn(x: np.ndarray, s: Optional[Tuple[int, ...]] = None,
         dim: Optional[Tuple[int, ...]] = None, norm: str = 'backward') -> np.ndarray:
    """
    Compute N-dimensional Discrete Fourier Transform.
    
    Args:
        x: Input tensor
        s: Shape of output for each dimension
        dim: Dimensions over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Complex tensor of N-D FFT coefficients
    """
    result = np.fft.fftn(x, s=s, axes=dim)
    if dim is None:
        n_total = x.size if s is None else np.prod(s)
    else:
        n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=True)


def ifftn(x: np.ndarray, s: Optional[Tuple[int, ...]] = None,
          dim: Optional[Tuple[int, ...]] = None, norm: str = 'backward') -> np.ndarray:
    """
    Compute N-dimensional Inverse Discrete Fourier Transform.
    
    Args:
        x: Input complex tensor
        s: Shape of output for each dimension
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Complex tensor of inverse N-D FFT
    """
    result = np.fft.ifftn(x, s=s, axes=dim)
    if dim is None:
        n_total = x.size if s is None else np.prod(s)
    else:
        n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=False)


def rfftn(x: np.ndarray, s: Optional[Tuple[int, ...]] = None,
          dim: Optional[Tuple[int, ...]] = None, norm: str = 'backward') -> np.ndarray:
    """
    Compute N-dimensional FFT of real-valued input.
    
    Args:
        x: Real input tensor
        s: Shape of output for each dimension
        dim: Dimensions over which to compute FFT
        norm: Normalization mode
        
    Returns:
        Complex tensor with positive frequencies
    """
    result = np.fft.rfftn(x, s=s, axes=dim)
    if dim is None:
        n_total = x.size if s is None else np.prod(s)
    else:
        n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=True)


def irfftn(x: np.ndarray, s: Optional[Tuple[int, ...]] = None,
           dim: Optional[Tuple[int, ...]] = None, norm: str = 'backward') -> np.ndarray:
    """
    Compute inverse N-dimensional FFT of real signal.
    
    Args:
        x: Complex input tensor
        s: Output shape for each dimension
        dim: Dimensions over which to compute IFFT
        norm: Normalization mode
        
    Returns:
        Real-valued tensor
    """
    result = np.fft.irfftn(x, s=s, axes=dim)
    if dim is None:
        n_total = x.size if s is None else np.prod(s)
    else:
        n_total = np.prod(s if s else [x.shape[d] for d in dim])
    return _apply_norm(result, n_total, norm, forward=False)


def hfftn(x: np.ndarray, s: Optional[Tuple[int, ...]] = None,
          dim: Optional[Tuple[int, ...]] = None, norm: str = 'backward') -> np.ndarray:
    """Compute N-D FFT of Hermitian symmetric signal."""
    return np.real(fftn(x, s=s, dim=dim, norm=norm))


def ihfftn(x: np.ndarray, s: Optional[Tuple[int, ...]] = None,
           dim: Optional[Tuple[int, ...]] = None, norm: str = 'backward') -> np.ndarray:
    """Compute inverse N-D FFT producing Hermitian symmetric output."""
    return ifftn(x.astype(complex), s=s, dim=dim, norm=norm)


# Frequency and Shift Operations

def fftfreq(n: int, d: float = 1.0, dtype: Optional[np.dtype] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Compute DFT sample frequencies.
    
    Args:
        n: Window length
        d: Sample spacing
        dtype: Output dtype
        device: Ignored (for API compatibility)
        
    Returns:
        Array of length n containing sample frequencies
    """
    result = np.fft.fftfreq(n, d=d)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def rfftfreq(n: int, d: float = 1.0, dtype: Optional[np.dtype] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Compute DFT sample frequencies for real FFT.
    
    Args:
        n: Window length
        d: Sample spacing
        dtype: Output dtype
        device: Ignored (for API compatibility)
        
    Returns:
        Array of length n//2+1 containing sample frequencies
    """
    result = np.fft.rfftfreq(n, d=d)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def fftshift(x: np.ndarray, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Shift zero-frequency component to center.
    
    Args:
        x: Input tensor
        dim: Dimensions over which to shift
        
    Returns:
        Shifted tensor
    """
    return np.fft.fftshift(x, axes=dim)


def ifftshift(x: np.ndarray, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Inverse of fftshift.
    
    Args:
        x: Input tensor
        dim: Dimensions over which to shift
        
    Returns:
        Shifted tensor
    """
    return np.fft.ifftshift(x, axes=dim)


# Short-Time Fourier Transform

def stft(x: np.ndarray, n_fft: int, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: Optional[np.ndarray] = None,
         center: bool = True, pad_mode: str = 'reflect', normalized: bool = False,
         onesided: bool = True, return_complex: bool = True) -> np.ndarray:
    """
    Short-Time Fourier Transform.
    
    Args:
        x: Input signal (..., time)
        n_fft: FFT size
        hop_length: Hop between windows (default: n_fft // 4)
        win_length: Window size (default: n_fft)
        window: Window function (default: Hann window)
        center: Pad signal on both sides
        pad_mode: Padding mode for center
        normalized: Apply normalization
        onesided: Return only positive frequencies for real input
        return_complex: Return complex or (real, imag) pair
        
    Returns:
        STFT tensor (..., freq, time)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = np.hanning(win_length)
    
    # Pad window to n_fft if needed
    if len(window) < n_fft:
        pad_left = (n_fft - len(window)) // 2
        pad_right = n_fft - len(window) - pad_left
        window = np.pad(window, (pad_left, pad_right))
    
    # Center padding
    if center:
        pad_amount = n_fft // 2
        if x.ndim == 1:
            x = np.pad(x, (pad_amount, pad_amount), mode=pad_mode)
        else:
            x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad_amount, pad_amount)], mode=pad_mode)
    
    # Compute number of frames
    n_frames = 1 + (x.shape[-1] - n_fft) // hop_length
    
    # Create output shape
    freq_bins = n_fft // 2 + 1 if onesided else n_fft
    batch_shape = x.shape[:-1]
    output = np.zeros(batch_shape + (freq_bins, n_frames), dtype=complex)
    
    # Compute STFT frame by frame
    for t in range(n_frames):
        start = t * hop_length
        frame = x[..., start:start + n_fft] * window
        if onesided:
            output[..., :, t] = np.fft.rfft(frame, n=n_fft, axis=-1)
        else:
            output[..., :, t] = np.fft.fft(frame, n=n_fft, axis=-1)
    
    if normalized:
        output = output / np.sqrt(n_fft)
    
    if return_complex:
        return output
    return np.stack([np.real(output), np.imag(output)], axis=-1)


def istft(x: np.ndarray, n_fft: int, hop_length: Optional[int] = None,
          win_length: Optional[int] = None, window: Optional[np.ndarray] = None,
          center: bool = True, normalized: bool = False, onesided: bool = True,
          length: Optional[int] = None, return_complex: bool = False) -> np.ndarray:
    """
    Inverse Short-Time Fourier Transform.
    
    Args:
        x: STFT tensor (..., freq, time)
        n_fft: FFT size
        hop_length: Hop between windows
        win_length: Window size
        window: Window function
        center: Whether input was center-padded
        normalized: Whether input was normalized
        onesided: Whether input contains only positive frequencies
        length: Expected output length
        return_complex: Return complex signal
        
    Returns:
        Reconstructed signal (..., time)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = np.hanning(win_length)
    
    # Pad window
    if len(window) < n_fft:
        pad_left = (n_fft - len(window)) // 2
        pad_right = n_fft - len(window) - pad_left
        window = np.pad(window, (pad_left, pad_right))
    
    if normalized:
        x = x * np.sqrt(n_fft)
    
    n_frames = x.shape[-1]
    expected_length = n_fft + hop_length * (n_frames - 1)
    
    batch_shape = x.shape[:-2]
    output = np.zeros(batch_shape + (expected_length,))
    window_sum = np.zeros(expected_length)
    
    for t in range(n_frames):
        start = t * hop_length
        if onesided:
            frame = np.fft.irfft(x[..., :, t], n=n_fft, axis=-1)
        else:
            frame = np.fft.ifft(x[..., :, t], n=n_fft, axis=-1)
        
        if not return_complex:
            frame = np.real(frame)
        
        output[..., start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2
    
    # Normalize by window sum
    window_sum = np.maximum(window_sum, 1e-8)
    output = output / window_sum
    
    # Remove center padding
    if center:
        output = output[..., n_fft // 2:]
    
    # Trim to length
    if length is not None:
        output = output[..., :length]
    
    return output


# Window Functions

def hann_window(window_length: int, periodic: bool = True, 
                dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate Hann window."""
    if periodic:
        window = np.hanning(window_length + 1)[:-1]
    else:
        window = np.hanning(window_length)
    if dtype is not None:
        window = window.astype(dtype)
    return window


def hamming_window(window_length: int, periodic: bool = True,
                   alpha: float = 0.54, beta: float = 0.46,
                   dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate Hamming window."""
    if periodic:
        n = np.arange(window_length + 1)
        window = alpha - beta * np.cos(2 * np.pi * n / window_length)
        window = window[:-1]
    else:
        n = np.arange(window_length)
        window = alpha - beta * np.cos(2 * np.pi * n / (window_length - 1))
    if dtype is not None:
        window = window.astype(dtype)
    return window


def blackman_window(window_length: int, periodic: bool = True,
                    dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate Blackman window."""
    if periodic:
        window = np.blackman(window_length + 1)[:-1]
    else:
        window = np.blackman(window_length)
    if dtype is not None:
        window = window.astype(dtype)
    return window


def bartlett_window(window_length: int, periodic: bool = True,
                    dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate Bartlett (triangular) window."""
    if periodic:
        window = np.bartlett(window_length + 1)[:-1]
    else:
        window = np.bartlett(window_length)
    if dtype is not None:
        window = window.astype(dtype)
    return window


def kaiser_window(window_length: int, beta: float = 12.0, periodic: bool = True,
                  dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate Kaiser window."""
    if periodic:
        window = np.kaiser(window_length + 1, beta)[:-1]
    else:
        window = np.kaiser(window_length, beta)
    if dtype is not None:
        window = window.astype(dtype)
    return window


def gaussian_window(window_length: int, std: float = 1.0,
                    dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate Gaussian window."""
    n = np.arange(window_length) - (window_length - 1) / 2
    window = np.exp(-0.5 * (n / (std * (window_length - 1) / 2)) ** 2)
    if dtype is not None:
        window = window.astype(dtype)
    return window


def cosine_window(window_length: int, periodic: bool = True,
                  dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Generate cosine window."""
    if periodic:
        n = np.arange(window_length + 1)
        window = np.sin(np.pi * n / window_length)[:-1]
    else:
        n = np.arange(window_length)
        window = np.sin(np.pi * n / (window_length - 1))
    if dtype is not None:
        window = window.astype(dtype)
    return window


# Helper Functions

def _apply_norm(x: np.ndarray, n: int, norm: str, forward: bool) -> np.ndarray:
    """Apply normalization to FFT result."""
    if norm == 'ortho':
        return x / np.sqrt(n)
    elif norm == 'forward':
        return x / n if forward else x
    else:  # 'backward' (default)
        return x if forward else x
    return x


# Export all functions
__all__ = [
    # 1D transforms
    'fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft',
    # 2D transforms
    'fft2', 'ifft2', 'rfft2', 'irfft2', 'hfft2', 'ihfft2',
    # N-D transforms
    'fftn', 'ifftn', 'rfftn', 'irfftn', 'hfftn', 'ihfftn',
    # Frequency and shift
    'fftfreq', 'rfftfreq', 'fftshift', 'ifftshift',
    # STFT
    'stft', 'istft',
    # Window functions
    'hann_window', 'hamming_window', 'blackman_window',
    'bartlett_window', 'kaiser_window', 'gaussian_window', 'cosine_window',
]
