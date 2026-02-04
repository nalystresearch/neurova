# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Time series analysis for Neurova."""

from neurova.timeseries.arima import ARIMA, auto_arima
from neurova.timeseries.exponential_smoothing import (
    SimpleExponentialSmoothing,
    ExponentialSmoothing,
)
from neurova.timeseries.decomposition import (
    seasonal_decompose,
    stl_decompose,
    DecompositionResult,
)
from neurova.timeseries.stattools import (
    acf,
    pacf,
    adfuller,
    ljung_box,
    ADFResult,
)

__all__ = [
    # aRIMA
    "ARIMA",
    "auto_arima",
    # exponential smoothing
    "SimpleExponentialSmoothing",
    "ExponentialSmoothing",
    # decomposition
    "seasonal_decompose",
    "stl_decompose",
    "DecompositionResult",
    # statistical tools
    "acf",
    "pacf",
    "adfuller",
    "ljung_box",
    "ADFResult",
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.