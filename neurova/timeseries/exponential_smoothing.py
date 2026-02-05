# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Exponential smoothing methods for time series."""

from __future__ import annotations
import numpy as np
from typing import Optional


class SimpleExponentialSmoothing:
    """Simple exponential smoothing for time series forecasting.
    
    Args:
        alpha: Smoothing parameter (0 < alpha < 1)
        
    Examples:
        model = SimpleExponentialSmoothing(alpha=0.3)
        model.fit(y)
        forecast = model.predict(steps=10)
    """
    
    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.is_fitted = False
        self.level: float = 0.0
        self.y: Optional[np.ndarray] = None
    
    def fit(self, y: np.ndarray):
        """Fit the model.
        
        Args:
            y: Time series data
            
        Returns:
            self
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        self.y = y
        
        # initialize level with first observation
        self.level = y[0]
        
        # update level through all observations
        for t in range(1, len(y)):
            self.level = self.alpha * y[t] + (1 - self.alpha) * self.level
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # sES forecast is constant at the last level
        return np.full(steps, self.level)


class ExponentialSmoothing:
    """Holt-Winters exponential smoothing (with trend and seasonality).
    
    Args:
        seasonal_periods: Number of periods in a season (e.g., 12 for monthly data with yearly seasonality)
        trend: Type of trend component ("add", "mul", or None)
        seasonal: Type of seasonal component ("add", "mul", or None)
        
    Examples:
        # additive model
        model = ExponentialSmoothing(seasonal_periods=12, trend="add", seasonal="add")
        model.fit(y)
        forecast = model.predict(steps=24)
    """
    
    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
    ):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.is_fitted = False
        
        # components
        self.level: float = 0.0
        self.trend_val: float = 0.0
        self.seasonal_vals: Optional[np.ndarray] = None
        
        # smoothing parameters (will be optimized)
        self.alpha: float = 0.3  # Level
        self.beta: float = 0.1   # Trend
        self.gamma: float = 0.1  # Seasonal
    
    def fit(self, y: np.ndarray):
        """Fit the model.
        
        Args:
            y: Time series data
            
        Returns:
            self
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        n = len(y)
        
        if self.seasonal and n < 2 * self.seasonal_periods:
            raise ValueError(f"Need at least {2 * self.seasonal_periods} observations for seasonal model")
        
        # initialize components
        self._initialize_components(y)
        
        # update components through all observations
        for t in range(len(y)):
            self._update_components(y[t], t)
        
        self.is_fitted = True
        return self
    
    def _initialize_components(self, y: np.ndarray):
        """Initialize level, trend, and seasonal components."""
        # level: average of first season
        if self.seasonal:
            self.level = float(np.mean(y[:self.seasonal_periods]))
        else:
            self.level = float(y[0])
        
        # trend: average difference between seasons
        if self.trend:
            if self.seasonal and len(y) >= 2 * self.seasonal_periods:
                season1 = np.mean(y[:self.seasonal_periods])
                season2 = np.mean(y[self.seasonal_periods:2*self.seasonal_periods])
                self.trend_val = float((season2 - season1) / self.seasonal_periods)
            else:
                self.trend_val = float((y[-1] - y[0]) / (len(y) - 1))
        else:
            self.trend_val = 0.0
        
        # seasonal: initial seasonal indices
        if self.seasonal:
            self.seasonal_vals = np.zeros(self.seasonal_periods)
            for i in range(self.seasonal_periods):
                # average value at this season across all years
                indices = list(range(i, len(y), self.seasonal_periods))
                if self.seasonal == "add":
                    self.seasonal_vals[i] = np.mean(y[indices]) - self.level
                else:  # multiplicative
                    self.seasonal_vals[i] = np.mean(y[indices]) / self.level if self.level != 0 else 1.0
        else:
            self.seasonal_vals = None
    
    def _update_components(self, y_t: float, t: int):
        """Update level, trend, and seasonal components."""
        # get seasonal component
        if self.seasonal and self.seasonal_vals is not None:
            s_idx = t % self.seasonal_periods
            s_t = self.seasonal_vals[s_idx]
        else:
            s_t = 0.0 if self.seasonal == "add" else 1.0
        
        # update level
        if self.seasonal == "add":
            level_new = self.alpha * (y_t - s_t) + (1 - self.alpha) * (self.level + self.trend_val)
        elif self.seasonal == "mul":
            level_new = self.alpha * (y_t / s_t if s_t != 0 else y_t) + (1 - self.alpha) * (self.level + self.trend_val)
        else:
            level_new = self.alpha * y_t + (1 - self.alpha) * (self.level + self.trend_val)
        
        # update trend
        if self.trend:
            trend_new = self.beta * (level_new - self.level) + (1 - self.beta) * self.trend_val
        else:
            trend_new = 0.0
        
        # update seasonal
        if self.seasonal and self.seasonal_vals is not None:
            s_idx = t % self.seasonal_periods
            if self.seasonal == "add":
                self.seasonal_vals[s_idx] = self.gamma * (y_t - level_new) + (1 - self.gamma) * s_t
            else:  # multiplicative
                self.seasonal_vals[s_idx] = self.gamma * (y_t / level_new if level_new != 0 else 1.0) + (1 - self.gamma) * s_t
        
        self.level = level_new
        self.trend_val = trend_new
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        forecast = np.zeros(steps)
        
        for h in range(steps):
            # base forecast: level + trend
            if self.trend:
                base = self.level + (h + 1) * self.trend_val
            else:
                base = self.level
            
            # add seasonal component
            if self.seasonal and self.seasonal_vals is not None:
                s_idx = h % self.seasonal_periods
                if self.seasonal == "add":
                    forecast[h] = base + self.seasonal_vals[s_idx]
                else:  # multiplicative
                    forecast[h] = base * self.seasonal_vals[s_idx]
            else:
                forecast[h] = base
        
        return forecast


__all__ = ["SimpleExponentialSmoothing", "ExponentialSmoothing"]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.