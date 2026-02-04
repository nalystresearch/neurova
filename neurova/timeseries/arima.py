# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""ARIMA (AutoRegressive Integrated Moving Average) implementation."""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from neurova.core.errors import ValidationError


class ARIMA:
    """ARIMA(p, d, q) model for time series forecasting.
    
    Args:
        order: (p, d, q) tuple where:
            p: autoregressive order
            d: differencing order
            q: moving average order
        seasonal_order: Optional (P, D, Q, s) for seasonal ARIMA
        
    Examples:
        # fit ARIMA(1,1,1)
        model = ARIMA(order=(1, 1, 1))
        model.fit(y)
        forecast = model.predict(steps=10)
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.p, self.d, self.q = order
        self.seasonal_order = seasonal_order
        self.is_fitted = False
        
        # parameters to be estimated
        self.ar_params: Optional[np.ndarray] = None
        self.ma_params: Optional[np.ndarray] = None
        self.const: float = 0.0
        
        # fitted data
        self.y_original: Optional[np.ndarray] = None
        self.y_diff: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
    
    def _difference(self, y: np.ndarray, d: int) -> np.ndarray:
        """Apply differencing to make series stationary."""
        y_diff = y.copy()
        for _ in range(d):
            y_diff = np.diff(y_diff)
        return y_diff
    
    def _undifference(self, y_diff: np.ndarray, y_original: np.ndarray, d: int) -> np.ndarray:
        """Reverse differencing to get back to original scale."""
        if d == 0:
            return y_diff
        
        y = y_diff.copy()
        for _ in range(d):
            # add back the last value from original series
            y = np.cumsum(np.concatenate([[y_original[-1]], y]))
        return y
    
    def fit(self, y: np.ndarray, max_iter: int = 100) -> ARIMA:
        """Fit ARIMA model to time series.
        
        Args:
            y: Time series data (1D array)
            max_iter: Maximum iterations for optimization
            
        Returns:
            self
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        if len(y) < max(self.p, self.q) + self.d + 10:
            raise ValidationError("y", len(y), f"at least {max(self.p, self.q) + self.d + 10} observations")
        
        self.y_original = y.copy()
        
        # apply differencing
        self.y_diff = self._difference(y, self.d)
        
        # fit AR and MA parameters using conditional least squares
        # this is a simplified implementation
        if self.p > 0 and self.q == 0:
            # pure AR model - use Yule-Walker equations
            self.ar_params, self.const = self._fit_ar(self.y_diff)
            self.ma_params = np.array([])
        elif self.q > 0 and self.p == 0:
            # pure MA model - use innovations algorithm
            self.ma_params, self.const = self._fit_ma(self.y_diff)
            self.ar_params = np.array([])
        else:
            # aRMA model - use conditional least squares
            self.ar_params, self.ma_params, self.const = self._fit_arma(self.y_diff)
        
        # calculate residuals
        self.residuals = self._calculate_residuals(self.y_diff)
        self.is_fitted = True
        
        return self
    
    def _fit_ar(self, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit AR(p) model using Yule-Walker equations."""
        # remove mean
        y_centered = y - y.mean()
        const = y.mean()
        
        # compute autocorrelations
        acf_vals = self._compute_acf(y_centered, self.p)
        
        # build Yule-Walker matrix
        R = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                R[i, j] = acf_vals[abs(i - j)]
        
        # solve for AR parameters
        r = acf_vals[1:self.p + 1]
        try:
            ar_params = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            # fallback to least squares
            ar_params = np.linalg.lstsq(R, r, rcond=None)[0]
        
        return ar_params, const
    
    def _fit_ma(self, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit MA(q) model using innovations algorithm (simplified)."""
        const = y.mean()
        y_centered = y - const
        
        # simplified MA estimation using autocorrelations
        acf_vals = self._compute_acf(y_centered, self.q)
        
        # approximate MA parameters from ACF
        # this is a simplified approach
        ma_params = -acf_vals[1:self.q + 1] * 0.5
        
        return ma_params, const
    
    def _fit_arma(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit ARMA(p,q) model using conditional least squares."""
        const = y.mean()
        y_centered = y - const
        
        # initialize parameters
        ar_params = np.random.randn(self.p) * 0.1
        ma_params = np.random.randn(self.q) * 0.1
        
        # simple gradient descent (very basic optimization)
        lr = 0.01
        for _ in range(50):
            errors = self._calculate_residuals_arma(y_centered, ar_params, ma_params)
            
            # update AR parameters
            for i in range(self.p):
                if len(y_centered) > i:
                    grad = -2 * np.mean(errors[self.p:] * y_centered[self.p - i - 1:-i - 1 if i > 0 else None])
                    ar_params[i] -= lr * grad
            
            # update MA parameters (simplified)
            for j in range(self.q):
                if len(errors) > j + 1:
                    grad = -2 * np.mean(errors[self.q:] * errors[self.q - j - 1:-j - 1 if j > 0 else None])
                    ma_params[j] -= lr * grad
        
        return ar_params, ma_params, const
    
    def _compute_acf(self, y: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function."""
        n = len(y)
        acf = np.zeros(max_lag + 1)
        
        var = np.var(y)
        if var == 0:
            return acf
        
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                acf[lag] = np.mean((y[:-lag] - y.mean()) * (y[lag:] - y.mean())) / var
        
        return acf
    
    def _calculate_residuals(self, y: np.ndarray) -> np.ndarray:
        """Calculate model residuals."""
        ar = self.ar_params if self.ar_params is not None else np.array([])
        ma = self.ma_params if self.ma_params is not None else np.array([])
        return self._calculate_residuals_arma(y - self.const, ar, ma)
    
    def _calculate_residuals_arma(
        self,
        y: np.ndarray,
        ar_params: np.ndarray,
        ma_params: np.ndarray,
    ) -> np.ndarray:
        """Calculate residuals for ARMA model."""
        n = len(y)
        errors = np.zeros(n)
        
        for t in range(max(self.p, self.q), n):
            # aR component
            ar_part = 0.0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_part += ar_params[i] * y[t - i - 1]
            
            # mA component
            ma_part = 0.0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_part += ma_params[j] * errors[t - j - 1]
            
            errors[t] = y[t] - ar_part - ma_part
        
        return errors
    
    def predict(self, steps: int = 1, return_conf_int: bool = False):
        """Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: If True, return confidence intervals
            
        Returns:
            Forecasted values (or tuple of (forecast, conf_int) if return_conf_int=True)
        """
        if not self.is_fitted or self.y_diff is None or self.residuals is None or self.y_original is None:
            raise RuntimeError("Model must be fitted before prediction")
        if self.ar_params is None or self.ma_params is None:
            raise RuntimeError("Model parameters not initialized")
        
        # forecast on differenced series
        forecast_diff = np.zeros(steps)
        
        # use last values for prediction
        last_values = self.y_diff[-max(self.p, self.q):].tolist()
        last_errors = self.residuals[-self.q:].tolist() if self.q > 0 else []
        
        for t in range(steps):
            # aR component
            ar_part = self.const
            for i in range(min(self.p, len(last_values))):
                ar_part += self.ar_params[i] * last_values[-(i + 1)]
            
            # mA component
            ma_part = 0.0
            for j in range(min(self.q, len(last_errors))):
                ma_part += self.ma_params[j] * last_errors[-(j + 1)]
            
            forecast_diff[t] = ar_part + ma_part
            
            # update history
            last_values.append(forecast_diff[t])
            last_errors.append(0.0)  # Future errors assumed to be 0
        
        # undifference if needed
        if self.d > 0:
            # simple undifferencing (cumulative sum from last known value)
            forecast = np.zeros(steps)
            last_val = self.y_original[-1]
            for t in range(steps):
                last_val = last_val + forecast_diff[t]
                forecast[t] = last_val
        else:
            forecast = forecast_diff
        
        if return_conf_int:
            # simple confidence interval based on residual variance
            std_error = np.std(self.residuals)
            conf_int = np.column_stack([
                forecast - 1.96 * std_error,
                forecast + 1.96 * std_error,
            ])
            return forecast, conf_int
        
        return forecast
    
    def aic(self) -> float:
        """Akaike Information Criterion."""
        if not self.is_fitted or self.residuals is None:
            raise RuntimeError("Model must be fitted first")
        
        n = len(self.residuals)
        k = self.p + self.q + 1  # +1 for constant
        sse = np.sum(self.residuals ** 2)
        
        return n * np.log(sse / n) + 2 * k
    
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        if not self.is_fitted or self.residuals is None:
            raise RuntimeError("Model must be fitted first")
        
        n = len(self.residuals)
        k = self.p + self.q + 1
        sse = np.sum(self.residuals ** 2)
        
        return n * np.log(sse / n) + k * np.log(n)


def auto_arima(
    y: np.ndarray,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    criterion: str = "aic",
) -> ARIMA:
    """Automatically select best ARIMA order.
    
    Args:
        y: Time series data
        max_p: Maximum AR order to try
        max_d: Maximum differencing order to try
        max_q: Maximum MA order to try
        criterion: Selection criterion ("aic" or "bic")
        
    Returns:
        Best ARIMA model
    """
    best_score = np.inf
    best_order = (0, 0, 0)
    best_model = None
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                
                try:
                    model = ARIMA(order=(p, d, q))
                    model.fit(y)
                    
                    score = model.aic() if criterion == "aic" else model.bic()
                    
                    if score < best_score:
                        best_score = score
                        best_order = (p, d, q)
                        best_model = model
                except Exception:
                    continue
    
    if best_model is None:
        # fallback to ARIMA(1,1,1)
        best_model = ARIMA(order=(1, 1, 1))
        best_model.fit(y)
    
    return best_model


__all__ = ["ARIMA", "auto_arima"]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.