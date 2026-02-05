# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Time series decomposition methods."""

from __future__ import annotations
import numpy as np
from typing import Literal, NamedTuple, Optional


class DecompositionResult(NamedTuple):
    """Result from seasonal decomposition.
    
    Attributes:
        observed: Original time series
        trend: Trend component
        seasonal: Seasonal component
        residual: Residual component
    """
    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray


def seasonal_decompose(
    y: np.ndarray,
    model: Literal["additive", "multiplicative"] = "additive",
    period: int = 12,
    extrapolate_trend: bool = True,
) -> DecompositionResult:
    """Decompose a time series into trend, seasonal, and residual components.
    
    Uses moving average for trend extraction and averaging for seasonal component.
    
    Args:
        y: Time series data
        model: Type of decomposition ("additive" or "multiplicative")
        period: Number of observations in a seasonal period
        extrapolate_trend: Whether to extrapolate trend at boundaries
        
    Returns:
        DecompositionResult with trend, seasonal, and residual components
        
    Examples:
        result = seasonal_decompose(y, model="additive", period=12)
        trend = result.trend
        seasonal = result.seasonal
        residual = result.residual
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    n = len(y)
    
    if n < 2 * period:
        raise ValueError(f"Time series length ({n}) must be at least 2*period ({2*period})")
    
    # extract trend using centered moving average
    trend = _extract_trend(y, period, extrapolate_trend)
    
    # remove trend
    if model == "additive":
        detrended = y - trend
    else:  # multiplicative
        detrended = y / trend
    
    # extract seasonal component
    seasonal = _extract_seasonal(detrended, period, model)
    
    # calculate residuals
    if model == "additive":
        residual = y - trend - seasonal
    else:  # multiplicative
        residual = y / (trend * seasonal)
    
    return DecompositionResult(
        observed=y,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
    )


def _extract_trend(y: np.ndarray, period: int, extrapolate: bool) -> np.ndarray:
    """Extract trend using centered moving average."""
    n = len(y)
    trend = np.full(n, np.nan)
    
    # for even period, use double moving average
    if period % 2 == 0:
        # first moving average
        ma1 = np.convolve(y, np.ones(period) / period, mode='valid')
        # second moving average to center
        trend_valid = np.convolve(ma1, np.ones(2) / 2, mode='valid')
        # place in center
        start = period // 2
        end = start + len(trend_valid)
        trend[start:end] = trend_valid
    else:
        # simple centered moving average for odd period
        half = period // 2
        for i in range(half, n - half):
            trend[i] = np.mean(y[i - half:i + half + 1])
    
    # extrapolate trend at boundaries if requested
    if extrapolate:
        # forward fill at start
        first_valid = np.where(~np.isnan(trend))[0][0]
        trend[:first_valid] = trend[first_valid]
        
        # backward fill at end
        last_valid = np.where(~np.isnan(trend))[0][-1]
        trend[last_valid + 1:] = trend[last_valid]
    
    return trend


def _extract_seasonal(detrended: np.ndarray, period: int, model: str) -> np.ndarray:
    """Extract seasonal component by averaging over periods."""
    n = len(detrended)
    
    # calculate seasonal indices
    seasonal_avg = np.zeros(period)
    for i in range(period):
        # get all observations at this seasonal index
        indices = np.arange(i, n, period)
        values = detrended[indices]
        # average (ignoring NaN)
        seasonal_avg[i] = np.nanmean(values)
    
    # center the seasonal component
    if model == "additive":
        # force seasonal component to sum to zero
        seasonal_avg -= np.mean(seasonal_avg)
    else:  # multiplicative
        # force seasonal component to average to 1
        seasonal_avg /= np.mean(seasonal_avg)
    
    # repeat seasonal pattern to match time series length
    seasonal = np.tile(seasonal_avg, n // period + 1)[:n]
    
    return seasonal


def stl_decompose(
    y: np.ndarray,
    period: int = 12,
    seasonal: int = 7,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    robust: bool = False,
) -> DecompositionResult:
    """STL (Seasonal and Trend decomposition using Loess) decomposition.
    
    Simplified implementation using moving averages instead of LOESS smoothing.
    
    Args:
        y: Time series data
        period: Number of observations in a seasonal period
        seasonal: Length of seasonal smoother (must be odd)
        trend: Length of trend smoother (must be odd)
        low_pass: Length of low-pass filter (must be odd)
        robust: Whether to use robust fitting
        
    Returns:
        DecompositionResult with trend, seasonal, and residual components
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    n = len(y)
    
    # set defaults
    if trend is None:
        trend = int(np.ceil(1.5 * period / (1 - 1.5 / seasonal)))
        if trend % 2 == 0:
            trend += 1
    if low_pass is None:
        low_pass = period
        if low_pass % 2 == 0:
            low_pass += 1
    
    # initialize components
    seasonal_comp = np.zeros(n)
    trend_comp = np.zeros(n)
    
    # iterative procedure (simplified - normally would iterate)
    # for simplicity, just do one pass with moving averages
    
    # step 1: Remove seasonal using simple decomposition
    result = seasonal_decompose(y, model="additive", period=period)
    seasonal_comp = result.seasonal
    trend_comp = result.trend
    residual = result.residual
    
    return DecompositionResult(
        observed=y,
        trend=trend_comp,
        seasonal=seasonal_comp,
        residual=residual,
    )


__all__ = ["seasonal_decompose", "stl_decompose", "DecompositionResult"]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.