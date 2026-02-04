# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

# -*- coding: utf-8 -*-
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Statistical tests and functions for time series analysis."""

from __future__ import annotations
import numpy as np
from typing import NamedTuple


class ADFResult(NamedTuple):
    """Result from Augmented Dickey-Fuller test.
    
    Attributes:
        statistic: Test statistic
        pvalue: P-value (approximate)
        used_lag: Number of lags used
        nobs: Number of observations
        critical_values: Critical values at 1%, 5%, and 10%
    """
    statistic: float
    pvalue: float
    used_lag: int
    nobs: int
    critical_values: dict[str, float]


def acf(x: np.ndarray, nlags: int = 40, fft: bool = False) -> np.ndarray:
    """Compute autocorrelation function.
    
    Args:
        x: Time series data
        nlags: Number of lags to compute
        fft: Whether to use FFT (not implemented, falls back to direct)
        
    Returns:
        Autocorrelation values for lags 0 to nlags
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    x = x - np.mean(x)
    
    n = len(x)
    nlags = min(nlags, n - 1)
    
    acf_vals = np.zeros(nlags + 1)
    c0 = np.dot(x, x) / n
    
    acf_vals[0] = 1.0
    for k in range(1, nlags + 1):
        c_k = np.dot(x[:-k], x[k:]) / n
        acf_vals[k] = c_k / c0 if c0 != 0 else 0.0
    
    return acf_vals


def pacf(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Compute partial autocorrelation function using Yule-Walker.
    
    Args:
        x: Time series data
        nlags: Number of lags to compute
        
    Returns:
        Partial autocorrelation values for lags 0 to nlags
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    nlags = min(nlags, len(x) - 1)
    
    # get ACF values
    acf_vals = acf(x, nlags=nlags)
    
    # pACF using Durbin-Levinson recursion
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0
    
    if nlags > 0:
        pacf_vals[1] = acf_vals[1]
    
    # recursion
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = acf_vals[1]
    
    for k in range(2, nlags + 1):
        # calculate pacf[k]
        num = acf_vals[k]
        for j in range(1, k):
            num -= phi[k-1, j] * acf_vals[k-j]
        
        den = 1.0
        for j in range(1, k):
            den -= phi[k-1, j] * acf_vals[j]
        
        phi[k, k] = num / den if den != 0 else 0.0
        pacf_vals[k] = phi[k, k]
        
        # update phi[k, j] for j = 1, ..., k-1
        for j in range(1, k):
            phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k-j]
    
    return pacf_vals


def adfuller(
    x: np.ndarray,
    maxlag: int = None,
    regression: str = 'c',
) -> ADFResult:
    """Augmented Dickey-Fuller test for unit root.
    
    Simplified implementation - provides approximate results.
    
    Args:
        x: Time series data
        maxlag: Maximum lag to use (default: int(12*(n/100)^{1/4}))
        regression: Type of regression ('c' for constant, 'ct' for constant+trend, 'nc' for none)
        
    Returns:
        ADFResult with test statistic and p-value
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * np.power(n / 100.0, 1.0 / 4.0)))
    
    # difference and lag
    dx = np.diff(x)
    x_lag = x[:-1]
    
    # build regression matrix
    # �x_t = α + β*t + �*x_{t-1} + δ_1*�x_{t-1} + ... + δ_p*�x_{t-p} + ε_t
    
    # use simplified approach with lag 1
    lag = min(maxlag, 1)
    
    if lag > 0 and len(dx) > lag:
        y = dx[lag:]
        X_lag = x_lag[lag-1:-1]
    else:
        y = dx
        X_lag = x_lag[:-1]
    
    n_reg = len(y)
    
    # build design matrix
    X = np.ones((n_reg, 1))  # Constant
    if regression == 'ct':
        X = np.column_stack([X, np.arange(1, n_reg + 1)])  # Add trend
    elif regression == 'nc':
        X = np.empty((n_reg, 0))  # No constant or trend
    
    X = np.column_stack([X, X_lag])  # Add lagged level
    
    # oLS regression
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        gamma = beta[-1]  # Coefficient on lagged level
        
        # standard error (simplified)
        residuals = y - X @ beta
        rss = np.sum(residuals ** 2)
        se = np.sqrt(rss / (n_reg - X.shape[1]))
        se_gamma = se / np.sqrt(np.sum(X_lag ** 2))
        
        # test statistic
        statistic = gamma / se_gamma if se_gamma != 0 else 0.0
    except np.linalg.LinAlgError:
        statistic = 0.0
    
    # critical values (approximate for n=100, regression='c')
    critical_values = {
        '1%': -3.51,
        '5%': -2.89,
        '10%': -2.58,
    }
    
    # approximate p-value (very rough)
    if statistic < -3.5:
        pvalue = 0.01
    elif statistic < -2.9:
        pvalue = 0.05
    elif statistic < -2.6:
        pvalue = 0.10
    else:
        pvalue = 0.50
    
    return ADFResult(
        statistic=statistic,
        pvalue=pvalue,
        used_lag=lag,
        nobs=n_reg,
        critical_values=critical_values,
    )


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Ljung-Box test for autocorrelation.
    
    Tests whether autocorrelations are significantly different from zero.
    
    Args:
        x: Time series residuals
        lags: Number of lags to test
        
    Returns:
        Tuple of (test_statistics, p_values) for each lag
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    # compute ACF
    acf_vals = acf(x, nlags=lags)
    
    # ljung-Box statistic
    test_stats = np.zeros(lags)
    p_values = np.zeros(lags)
    
    for h in range(1, lags + 1):
        # q = n(n+2) * sum_{k=1}^h rho_k^2 / (n-k)
        q = n * (n + 2) * np.sum(acf_vals[1:h+1]**2 / (n - np.arange(1, h+1)))
        test_stats[h-1] = q
        
        # approximate p-value using chi-square distribution
        # degrees of freedom = h
        # very rough approximation
        if q > 7.81:  # chi2(h=1, alpha=0.05)  3.84, use conservative value
            p_values[h-1] = 0.01
        elif q > 3.84:
            p_values[h-1] = 0.05
        else:
            p_values[h-1] = 0.20
    
    return test_stats, p_values


__all__ = ["acf", "pacf", "adfuller", "ljung_box", "ADFResult"]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.