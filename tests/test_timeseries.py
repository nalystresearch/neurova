# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Tests for time series analysis module."""

import numpy as np
from neurova.timeseries import (
    ARIMA,
    auto_arima,
    SimpleExponentialSmoothing,
    ExponentialSmoothing,
    seasonal_decompose,
    acf,
    pacf,
    adfuller,
)


def test_arima_basic():
    """Test basic ARIMA functionality."""
    print("Testing ARIMA...")
    
    # generate AR(1) process: y_t = 0.5*y_{t-1} + noise
    np.random.seed(42)
    n = 100
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t-1] + np.random.randn()
    
    # fit ARIMA(1,0,0)
    model = ARIMA(order=(1, 0, 0))
    model.fit(y)
    
    assert model.is_fitted, "Model should be fitted"
    assert model.ar_params is not None, "AR parameters should exist"
    print(f"  AR parameter: {model.ar_params[0]:.3f} (expected ~0.5)")
    
    # predict
    forecast = model.predict(steps=10)
    assert len(forecast) == 10, "Should forecast 10 steps"
    print(f"  Forecast shape: {forecast.shape}")
    
    # check AIC/BIC
    aic = model.aic()
    bic = model.bic()
    print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}")
    
    print(" ARIMA test passed\n")


def test_auto_arima():
    """Test automatic ARIMA order selection."""
    print("Testing auto_arima...")
    
    # generate simple time series
    np.random.seed(42)
    y = np.cumsum(np.random.randn(50))  # Random walk
    
    # auto select order
    best_model = auto_arima(y, max_p=2, max_q=2, max_d=1)
    
    assert best_model.is_fitted, "Best model should be fitted"
    print(f"  Selected order: ({best_model.p}, {best_model.d}, {best_model.q})")
    print(f"  AIC: {best_model.aic():.2f}")
    
    print(" auto_arima test passed\n")


def test_exponential_smoothing():
    """Test exponential smoothing."""
    print("Testing Exponential Smoothing...")
    
    # generate trend + noise
    np.random.seed(42)
    t = np.arange(50)
    y = 10 + 0.5 * t + np.random.randn(50) * 0.5
    
    # simple ES
    ses = SimpleExponentialSmoothing(alpha=0.3)
    ses.fit(y)
    forecast_ses = ses.predict(steps=10)
    print(f"  SES forecast (constant): {forecast_ses[0]:.2f}")
    
    # holt (with trend)
    holt = ExponentialSmoothing(trend="add", seasonal=None)
    holt.fit(y)
    forecast_holt = holt.predict(steps=10)
    print(f"  Holt forecast: {forecast_holt[:3]}")
    
    print(" Exponential smoothing test passed\n")


def test_seasonal_decomposition():
    """Test seasonal decomposition."""
    print("Testing seasonal decomposition...")
    
    # generate seasonal data: trend + seasonal + noise
    np.random.seed(42)
    n = 72  # 6 years of monthly data
    t = np.arange(n)
    trend = 100 + 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(n) * 2
    y = trend + seasonal + noise
    
    # decompose
    result = seasonal_decompose(y, model="additive", period=12)
    
    assert len(result.trend) == n, "Trend should have same length as data"
    assert len(result.seasonal) == n, "Seasonal should have same length as data"
    assert len(result.residual) == n, "Residual should have same length as data"
    
    # check that seasonal component averages to ~0
    seasonal_mean = np.mean(result.seasonal)
    print(f"  Seasonal component mean: {seasonal_mean:.3f} (should be ~0)")
    assert abs(seasonal_mean) < 0.1, "Seasonal should average to 0"
    
    # check reconstruction
    reconstructed = result.trend + result.seasonal + result.residual
    reconstruction_error = np.max(np.abs(reconstructed - y))
    print(f"  Reconstruction error: {reconstruction_error:.6f}")
    assert reconstruction_error < 1e-10, "Should reconstruct perfectly"
    
    print(" Seasonal decomposition test passed\n")


def test_acf_pacf():
    """Test autocorrelation functions."""
    print("Testing ACF/PACF...")
    
    # generate AR(1) process
    np.random.seed(42)
    n = 100
    phi = 0.7
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.randn()
    
    # compute ACF
    acf_vals = acf(y, nlags=10)
    assert len(acf_vals) == 11, "Should have 11 ACF values (0 to 10)"
    assert abs(acf_vals[0] - 1.0) < 1e-10, "ACF at lag 0 should be 1"
    print(f"  ACF[1]: {acf_vals[1]:.3f} (expected ~{phi:.3f})")
    
    # compute PACF
    pacf_vals = pacf(y, nlags=10)
    assert len(pacf_vals) == 11, "Should have 11 PACF values"
    print(f"  PACF[1]: {pacf_vals[1]:.3f} (expected ~{phi:.3f})")
    print(f"  PACF[2]: {pacf_vals[2]:.3f} (expected ~0)")
    
    print(" ACF/PACF test passed\n")


def test_stationarity():
    """Test stationarity test."""
    print("Testing ADF stationarity test...")
    
    # stationary series (white noise)
    np.random.seed(42)
    stationary = np.random.randn(100)
    
    result = adfuller(stationary)
    print(f"  Stationary series - ADF statistic: {result.statistic:.3f}")
    print(f"  P-value: {result.pvalue:.3f}")
    print(f"  Critical values: {result.critical_values}")
    
    # non-stationary series (random walk)
    non_stationary = np.cumsum(np.random.randn(100))
    result2 = adfuller(non_stationary)
    print(f"  Non-stationary series - ADF statistic: {result2.statistic:.3f}")
    print(f"  P-value: {result2.pvalue:.3f}")
    
    print(" ADF test passed\n")


def test_seasonal_forecast():
    """Test seasonal forecasting with Holt-Winters."""
    print("Testing Holt-Winters seasonal forecasting...")
    
    # generate seasonal data with trend
    np.random.seed(42)
    n = 48  # 4 years of quarterly data
    t = np.arange(n)
    trend = 100 + 2 * t
    seasonal_pattern = np.array([10, -5, -10, 5])  # Quarterly pattern
    seasonal = np.tile(seasonal_pattern, n // 4)
    noise = np.random.randn(n) * 3
    y = trend + seasonal + noise
    
    # fit Holt-Winters
    hw = ExponentialSmoothing(seasonal_periods=4, trend="add", seasonal="add")
    hw.fit(y)
    
    # forecast
    forecast = hw.predict(steps=8)  # 2 years ahead
    print(f"  Forecast for next 8 quarters: {forecast}")
    print(f"  Forecast mean: {np.mean(forecast):.2f}")
    
    # check that forecast captures trend
    assert forecast[4] > forecast[0], "Should show increasing trend"
    
    print(" Holt-Winters test passed\n")


def run_all_tests():
    """Run all time series tests."""
    print("="*60)
    print("NEUROVA TIME SERIES TESTS")
    print("="*60 + "\n")
    
    test_arima_basic()
    test_auto_arima()
    test_exponential_smoothing()
    test_seasonal_decomposition()
    test_acf_pacf()
    test_stationarity()
    test_seasonal_forecast()
    
    print("="*60)
    print("ALL TESTS PASSED ")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
