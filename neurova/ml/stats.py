# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Statistical hypothesis tests."""

from __future__ import annotations
import numpy as np
from typing import NamedTuple

try:
    from scipy import special as sp_special
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TTestResult(NamedTuple):
    """Result from t-test."""
    statistic: float
    pvalue: float
    df: int


class FTestResult(NamedTuple):
    """Result from F-test (ANOVA)."""
    statistic: float
    pvalue: float
    df_between: int
    df_within: int


class ChiSquareResult(NamedTuple):
    """Result from chi-square test."""
    statistic: float
    pvalue: float
    df: int
    expected_freq: np.ndarray


def ttest_ind(a: np.ndarray, b: np.ndarray, equal_var: bool = True) -> TTestResult:
    """Independent samples t-test.
    
    Args:
        a: First sample
        b: Second sample
        equal_var: Whether to assume equal variances
        
    Returns:
        TTestResult with statistic, p-value, and degrees of freedom
        
    Examples:
        result = ttest_ind(group1, group2)
        print(f"t-statistic: {result.statistic}, p-value: {result.pvalue}")
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    n1 = len(a)
    n2 = len(b)
    
    mean1 = np.mean(a)
    mean2 = np.mean(b)
    
    var1 = np.var(a, ddof=1)
    var2 = np.var(b, ddof=1)
    
    if equal_var:
        # pooled variance
        df = n1 + n2 - 2
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
        se = np.sqrt(pooled_var * (1/n1 + 1/n2))
    else:
        # welch's t-test
        se = np.sqrt(var1/n1 + var2/n2)
        # welch-Satterthwaite degrees of freedom
        df = int((var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)))
    
    t_stat = (mean1 - mean2) / (se + 1e-10)
    
    # approximate p-value using t-distribution
    pvalue = _t_pvalue(t_stat, df)
    
    return TTestResult(statistic=t_stat, pvalue=pvalue, df=df)


def ttest_1samp(a: np.ndarray, popmean: float) -> TTestResult:
    """One-sample t-test.
    
    Args:
        a: Sample data
        popmean: Expected population mean
        
    Returns:
        TTestResult
    """
    a = np.asarray(a).flatten()
    n = len(a)
    df = n - 1
    
    mean = np.mean(a)
    std = np.std(a, ddof=1)
    se = std / np.sqrt(n)
    
    t_stat = (mean - popmean) / (se + 1e-10)
    pvalue = _t_pvalue(t_stat, df)
    
    return TTestResult(statistic=t_stat, pvalue=pvalue, df=df)


def ttest_rel(a: np.ndarray, b: np.ndarray) -> TTestResult:
    """Paired (related) samples t-test.
    
    Args:
        a: First sample
        b: Second sample (must have same length)
        
    Returns:
        TTestResult
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    
    if len(a) != len(b):
        raise ValueError("Samples must have equal length")
    
    diff = a - b
    return ttest_1samp(diff, 0.0)


def f_oneway(*groups: np.ndarray) -> FTestResult:
    """One-way ANOVA F-test.
    
    Args:
        *groups: Variable number of sample arrays
        
    Returns:
        FTestResult with F-statistic and p-value
        
    Examples:
        result = f_oneway(group1, group2, group3)
        print(f"F-statistic: {result.statistic}, p-value: {result.pvalue}")
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    
    groups = [np.asarray(g).flatten() for g in groups]
    
    # grand mean
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    n_total = len(all_data)
    k = len(groups)  # Number of groups
    
    # between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    df_between = k - 1
    
    # within-group sum of squares
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
    df_within = n_total - k
    
    # mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / (df_within + 1e-10)
    
    # f-statistic
    f_stat = ms_between / (ms_within + 1e-10)
    
    # approximate p-value
    pvalue = _f_pvalue(f_stat, df_between, df_within)
    
    return FTestResult(
        statistic=f_stat,
        pvalue=pvalue,
        df_between=df_between,
        df_within=df_within,
    )


def chi2_contingency(observed: np.ndarray) -> ChiSquareResult:
    """Chi-square test of independence for contingency table.
    
    Args:
        observed: Contingency table (2D array)
        
    Returns:
        ChiSquareResult with chi-square statistic and p-value
        
    Examples:
        # 2x2 contingency table
        table = np.array([[10, 20], [30, 40]])
        result = chi2_contingency(table)
    """
    observed = np.asarray(observed)
    
    if observed.ndim != 2:
        raise ValueError("Observed must be 2D array")
    
    # calculate expected frequencies
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    
    expected = (row_sums @ col_sums) / (total + 1e-10)
    
    # chi-square statistic
    chi2_stat = np.sum((observed - expected)**2 / (expected + 1e-10))
    
    # degrees of freedom
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    
    # approximate p-value
    pvalue = _chi2_pvalue(chi2_stat, df)
    
    return ChiSquareResult(
        statistic=chi2_stat,
        pvalue=pvalue,
        df=df,
        expected_freq=expected,
    )


def kstest(data: np.ndarray, distribution: str = "norm") -> tuple[float, float]:
    """Kolmogorov-Smirnov test for goodness of fit.
    
    Args:
        data: Sample data
        distribution: Distribution to test against ("norm" for normal)
        
    Returns:
        Tuple of (statistic, pvalue)
    """
    data = np.asarray(data).flatten()
    data_sorted = np.sort(data)
    n = len(data)
    
    if distribution == "norm":
        # standardize data
        mean = np.mean(data)
        std = np.std(data)
        data_std = (data_sorted - mean) / (std + 1e-10)
        
        # empirical CDF
        ecdf = np.arange(1, n + 1) / n
        
        # theoretical CDF (normal)
        tcdf = _norm_cdf(data_std)
        
        # kS statistic
        ks_stat = np.max(np.abs(ecdf - tcdf))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # approximate p-value
    pvalue = _ks_pvalue(ks_stat, n)
    
    return ks_stat, pvalue


def _t_pvalue(t_stat: float, df: int) -> float:
    """Approximate p-value for t-distribution (two-tailed)."""
    # very rough approximation
    abs_t = abs(t_stat)
    
    if df < 1:
        return 1.0
    
    # critical values for common df
    if df >= 30:
        # use normal approximation
        if abs_t > 2.58:
            return 0.01
        elif abs_t > 1.96:
            return 0.05
        elif abs_t > 1.65:
            return 0.10
        else:
            return 0.20
    else:
        # use conservative t-values
        if abs_t > 3.0:
            return 0.01
        elif abs_t > 2.0:
            return 0.05
        elif abs_t > 1.5:
            return 0.15
        else:
            return 0.30


def _f_pvalue(f_stat: float, df1: int, df2: int) -> float:
    """Approximate p-value for F-distribution."""
    if f_stat < 1.0:
        return 0.90
    elif f_stat < 2.0:
        return 0.20
    elif f_stat < 3.0:
        return 0.10
    elif f_stat < 4.0:
        return 0.05
    elif f_stat < 6.0:
        return 0.02
    else:
        return 0.01


def _chi2_pvalue(chi2_stat: float, df: int) -> float:
    """Approximate p-value for chi-square distribution."""
    if df < 1:
        return 1.0
    
    # critical values approximation
    critical_values = {
        1: [2.71, 3.84, 6.63, 10.83],  # p = [0.10, 0.05, 0.01, 0.001]
        2: [4.61, 5.99, 9.21, 13.82],
        3: [6.25, 7.81, 11.34, 16.27],
        4: [7.78, 9.49, 13.28, 18.47],
        5: [9.24, 11.07, 15.09, 20.52],
    }
    
    df_key = min(df, 5)
    crits = critical_values.get(df_key, critical_values[5])
    
    if chi2_stat < crits[0]:
        return 0.20
    elif chi2_stat < crits[1]:
        return 0.08
    elif chi2_stat < crits[2]:
        return 0.02
    elif chi2_stat < crits[3]:
        return 0.005
    else:
        return 0.001


def _ks_pvalue(ks_stat: float, n: int) -> float:
    """Approximate p-value for KS test."""
    # kolmogorov distribution approximation
    lambda_val = (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n)) * ks_stat
    
    if lambda_val < 0.5:
        return 1.0
    elif lambda_val < 1.0:
        return 0.30
    elif lambda_val < 1.36:
        return 0.05
    elif lambda_val < 1.63:
        return 0.01
    else:
        return 0.001


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Cumulative distribution function for standard normal."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Error function approximation."""
    # abramowitz and Stegun approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    sign = np.sign(x)
    x = np.abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y


__all__ = [
    "ttest_ind",
    "ttest_1samp",
    "ttest_rel",
    "f_oneway",
    "chi2_contingency",
    "kstest",
    "TTestResult",
    "FTestResult",
    "ChiSquareResult",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.