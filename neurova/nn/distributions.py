# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Probability Distributions.

Neurova implementation of probability distributions for
probabilistic modeling and sampling.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    Base class for probability distributions.
    
    Neurova implementation.
    
    All distributions support sampling and probability computation.
    """
    
    has_rsample = False
    has_enumerate_support = False
    
    @abstractmethod
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Generate a sample from the distribution."""
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability of a value."""
        raise NotImplementedError
    
    def prob(self, value: np.ndarray) -> np.ndarray:
        """Compute probability of a value."""
        return np.exp(self.log_prob(value))
    
    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Distribution mean."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def variance(self) -> np.ndarray:
        """Distribution variance."""
        raise NotImplementedError
    
    @property
    def stddev(self) -> np.ndarray:
        """Distribution standard deviation."""
        return np.sqrt(self.variance)
    
    def entropy(self) -> np.ndarray:
        """Compute entropy of the distribution."""
        raise NotImplementedError
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        raise NotImplementedError
    
    def icdf(self, value: np.ndarray) -> np.ndarray:
        """Inverse cumulative distribution function (quantile)."""
        raise NotImplementedError


class Normal(Distribution):
    """
    Normal (Gaussian) distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : float or array_like
        Mean of the distribution (mu)
    scale : float or array_like
        Standard deviation of the distribution (sigma)
    
    Examples
    --------
    >>> dist = Normal(loc=0.0, scale=1.0)
    >>> samples = dist.sample((100,))
    >>> log_p = dist.log_prob(samples)
    """
    
    def __init__(self, loc: Union[float, np.ndarray], scale: Union[float, np.ndarray]):
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from normal distribution."""
        shape = sample_shape + self.loc.shape
        return np.random.normal(self.loc, self.scale, shape).astype(np.float32)
    
    def rsample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Reparameterized sample (differentiable)."""
        eps = np.random.standard_normal(sample_shape + self.loc.shape).astype(np.float32)
        return self.loc + self.scale * eps
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        var = self.scale ** 2
        log_scale = np.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - 0.5 * np.log(2 * np.pi)
    
    @property
    def mean(self) -> np.ndarray:
        return self.loc
    
    @property
    def variance(self) -> np.ndarray:
        return self.scale ** 2
    
    def entropy(self) -> np.ndarray:
        """Entropy of normal distribution."""
        return 0.5 + 0.5 * np.log(2 * np.pi) + np.log(self.scale)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        from scipy import special
        return 0.5 * (1 + special.erf((value - self.loc) / (self.scale * np.sqrt(2))))
    
    def icdf(self, value: np.ndarray) -> np.ndarray:
        """Inverse CDF (quantile function)."""
        from scipy import special
        return self.loc + self.scale * np.sqrt(2) * special.erfinv(2 * value - 1)


class Uniform(Distribution):
    """
    Uniform distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    low : float or array_like
        Lower bound
    high : float or array_like
        Upper bound
    
    Examples
    --------
    >>> dist = Uniform(low=0.0, high=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(self, low: Union[float, np.ndarray], high: Union[float, np.ndarray]):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from uniform distribution."""
        shape = sample_shape + self.low.shape
        return np.random.uniform(self.low, self.high, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        lb = self.low <= value
        ub = value < self.high
        return np.where(lb & ub, -np.log(self.high - self.low), -np.inf)
    
    @property
    def mean(self) -> np.ndarray:
        return (self.low + self.high) / 2
    
    @property
    def variance(self) -> np.ndarray:
        return (self.high - self.low) ** 2 / 12
    
    def entropy(self) -> np.ndarray:
        """Entropy of uniform distribution."""
        return np.log(self.high - self.low)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        result = (value - self.low) / (self.high - self.low)
        return np.clip(result, 0, 1)


class Bernoulli(Distribution):
    """
    Bernoulli distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    probs : float or array_like, optional
        Probability of success
    logits : float or array_like, optional
        Log-odds of success
    
    Examples
    --------
    >>> dist = Bernoulli(probs=0.5)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        probs: Optional[Union[float, np.ndarray]] = None,
        logits: Optional[Union[float, np.ndarray]] = None
    ):
        if probs is not None and logits is not None:
            raise ValueError("Only one of probs or logits should be specified")
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        
        if probs is not None:
            self.probs = np.asarray(probs, dtype=np.float32)
            self.logits = np.log(self.probs / (1 - self.probs))
        else:
            self.logits = np.asarray(logits, dtype=np.float32)
            self.probs = 1 / (1 + np.exp(-self.logits))
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Bernoulli distribution."""
        shape = sample_shape + self.probs.shape
        return (np.random.random(shape) < self.probs).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        return value * np.log(self.probs) + (1 - value) * np.log(1 - self.probs)
    
    @property
    def mean(self) -> np.ndarray:
        return self.probs
    
    @property
    def variance(self) -> np.ndarray:
        return self.probs * (1 - self.probs)
    
    def entropy(self) -> np.ndarray:
        """Entropy of Bernoulli distribution."""
        p = self.probs
        return -p * np.log(p) - (1 - p) * np.log(1 - p)


class Categorical(Distribution):
    """
    Categorical distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    probs : array_like, optional
        Probability of each category
    logits : array_like, optional
        Log-odds of each category
    
    Examples
    --------
    >>> dist = Categorical(probs=[0.25, 0.25, 0.25, 0.25])
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        probs: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None
    ):
        if probs is not None and logits is not None:
            raise ValueError("Only one of probs or logits should be specified")
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        
        if probs is not None:
            self.probs = np.asarray(probs, dtype=np.float32)
            self.logits = np.log(self.probs)
        else:
            self.logits = np.asarray(logits, dtype=np.float32)
            # softmax
            exp_logits = np.exp(self.logits - np.max(self.logits, axis=-1, keepdims=True))
            self.probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    @property
    def num_categories(self) -> int:
        return self.probs.shape[-1]
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from categorical distribution."""
        num_samples = int(np.prod(sample_shape)) if sample_shape else 1
        samples = np.array([
            np.random.choice(self.num_categories, p=self.probs.flatten())
            for _ in range(num_samples)
        ])
        return samples.reshape(sample_shape) if sample_shape else samples[0]
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        value = np.asarray(value, dtype=np.int64)
        return np.log(self.probs[value])
    
    @property
    def mean(self) -> np.ndarray:
        return np.sum(np.arange(self.num_categories) * self.probs)
    
    @property
    def variance(self) -> np.ndarray:
        mean = self.mean
        return np.sum(((np.arange(self.num_categories) - mean) ** 2) * self.probs)
    
    def entropy(self) -> np.ndarray:
        """Entropy of categorical distribution."""
        return -np.sum(self.probs * np.log(self.probs + 1e-8))


class Exponential(Distribution):
    """
    Exponential distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    rate : float or array_like
        Rate parameter (lambda)
    
    Examples
    --------
    >>> dist = Exponential(rate=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(self, rate: Union[float, np.ndarray]):
        self.rate = np.asarray(rate, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from exponential distribution."""
        shape = sample_shape + self.rate.shape
        return np.random.exponential(1 / self.rate, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        return np.log(self.rate) - self.rate * value
    
    @property
    def mean(self) -> np.ndarray:
        return 1 / self.rate
    
    @property
    def variance(self) -> np.ndarray:
        return 1 / (self.rate ** 2)
    
    def entropy(self) -> np.ndarray:
        """Entropy of exponential distribution."""
        return 1 - np.log(self.rate)
    
    def cdf(self, value: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        return 1 - np.exp(-self.rate * value)


class Gamma(Distribution):
    """
    Gamma distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    concentration : float or array_like
        Shape parameter (alpha)
    rate : float or array_like
        Rate parameter (beta)
    
    Examples
    --------
    >>> dist = Gamma(concentration=2.0, rate=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        concentration: Union[float, np.ndarray],
        rate: Union[float, np.ndarray]
    ):
        self.concentration = np.asarray(concentration, dtype=np.float32)
        self.rate = np.asarray(rate, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from gamma distribution."""
        shape = sample_shape + self.concentration.shape
        return np.random.gamma(self.concentration, 1 / self.rate, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        alpha, beta = self.concentration, self.rate
        return (alpha * np.log(beta) - special.gammaln(alpha) +
                (alpha - 1) * np.log(value) - beta * value)
    
    @property
    def mean(self) -> np.ndarray:
        return self.concentration / self.rate
    
    @property
    def variance(self) -> np.ndarray:
        return self.concentration / (self.rate ** 2)
    
    def entropy(self) -> np.ndarray:
        """Entropy of gamma distribution."""
        from scipy import special
        alpha, beta = self.concentration, self.rate
        return (alpha - np.log(beta) + special.gammaln(alpha) +
                (1 - alpha) * special.digamma(alpha))


class Beta(Distribution):
    """
    Beta distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    concentration1 : float or array_like
        First concentration parameter (alpha)
    concentration0 : float or array_like
        Second concentration parameter (beta)
    
    Examples
    --------
    >>> dist = Beta(concentration1=2.0, concentration0=5.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        concentration1: Union[float, np.ndarray],
        concentration0: Union[float, np.ndarray]
    ):
        self.concentration1 = np.asarray(concentration1, dtype=np.float32)  # alpha
        self.concentration0 = np.asarray(concentration0, dtype=np.float32)  # beta
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from beta distribution."""
        shape = sample_shape + self.concentration1.shape
        return np.random.beta(
            self.concentration1, self.concentration0, shape
        ).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        a, b = self.concentration1, self.concentration0
        return ((a - 1) * np.log(value) + (b - 1) * np.log(1 - value) -
                special.betaln(a, b))
    
    @property
    def mean(self) -> np.ndarray:
        return self.concentration1 / (self.concentration1 + self.concentration0)
    
    @property
    def variance(self) -> np.ndarray:
        a, b = self.concentration1, self.concentration0
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def entropy(self) -> np.ndarray:
        """Entropy of beta distribution."""
        from scipy import special
        a, b = self.concentration1, self.concentration0
        return (special.betaln(a, b) - (a - 1) * special.digamma(a) -
                (b - 1) * special.digamma(b) + (a + b - 2) * special.digamma(a + b))


class Poisson(Distribution):
    """
    Poisson distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    rate : float or array_like
        Rate parameter (lambda)
    
    Examples
    --------
    >>> dist = Poisson(rate=5.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(self, rate: Union[float, np.ndarray]):
        self.rate = np.asarray(rate, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Poisson distribution."""
        shape = sample_shape + self.rate.shape
        return np.random.poisson(self.rate, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        return value * np.log(self.rate) - self.rate - special.gammaln(value + 1)
    
    @property
    def mean(self) -> np.ndarray:
        return self.rate
    
    @property
    def variance(self) -> np.ndarray:
        return self.rate


class Binomial(Distribution):
    """
    Binomial distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    total_count : int or array_like
        Number of trials
    probs : float or array_like, optional
        Probability of success
    logits : float or array_like, optional
        Log-odds of success
    
    Examples
    --------
    >>> dist = Binomial(total_count=10, probs=0.5)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        total_count: Union[int, np.ndarray],
        probs: Optional[Union[float, np.ndarray]] = None,
        logits: Optional[Union[float, np.ndarray]] = None
    ):
        self.total_count = np.asarray(total_count, dtype=np.int64)
        
        if probs is not None and logits is not None:
            raise ValueError("Only one of probs or logits should be specified")
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        
        if probs is not None:
            self.probs = np.asarray(probs, dtype=np.float32)
            self.logits = np.log(self.probs / (1 - self.probs))
        else:
            self.logits = np.asarray(logits, dtype=np.float32)
            self.probs = 1 / (1 + np.exp(-self.logits))
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from binomial distribution."""
        shape = sample_shape + self.probs.shape
        return np.random.binomial(self.total_count, self.probs, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        n, k, p = self.total_count, value, self.probs
        return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1) +
                k * np.log(p) + (n - k) * np.log(1 - p))
    
    @property
    def mean(self) -> np.ndarray:
        return self.total_count * self.probs
    
    @property
    def variance(self) -> np.ndarray:
        return self.total_count * self.probs * (1 - self.probs)


class Dirichlet(Distribution):
    """
    Dirichlet distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    concentration : array_like
        Concentration parameters (alpha)
    
    Examples
    --------
    >>> dist = Dirichlet(concentration=[1.0, 2.0, 3.0])
    >>> samples = dist.sample((100,))
    """
    
    def __init__(self, concentration: np.ndarray):
        self.concentration = np.asarray(concentration, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Dirichlet distribution."""
        shape = sample_shape
        return np.random.dirichlet(self.concentration, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        alpha = self.concentration
        return (special.gammaln(np.sum(alpha)) - np.sum(special.gammaln(alpha)) +
                np.sum((alpha - 1) * np.log(value), axis=-1))
    
    @property
    def mean(self) -> np.ndarray:
        return self.concentration / np.sum(self.concentration)
    
    @property
    def variance(self) -> np.ndarray:
        alpha0 = np.sum(self.concentration)
        return (self.concentration * (alpha0 - self.concentration) /
                (alpha0 ** 2 * (alpha0 + 1)))
    
    def entropy(self) -> np.ndarray:
        """Entropy of Dirichlet distribution."""
        from scipy import special
        alpha = self.concentration
        alpha0 = np.sum(alpha)
        k = len(alpha)
        return (special.gammaln(alpha0) - np.sum(special.gammaln(alpha)) +
                (alpha0 - k) * special.digamma(alpha0) -
                np.sum((alpha - 1) * special.digamma(alpha)))


class Laplace(Distribution):
    """
    Laplace distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : float or array_like
        Location parameter (mu)
    scale : float or array_like
        Scale parameter (b)
    
    Examples
    --------
    >>> dist = Laplace(loc=0.0, scale=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        loc: Union[float, np.ndarray],
        scale: Union[float, np.ndarray]
    ):
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Laplace distribution."""
        shape = sample_shape + self.loc.shape
        return np.random.laplace(self.loc, self.scale, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        return -np.log(2 * self.scale) - np.abs(value - self.loc) / self.scale
    
    @property
    def mean(self) -> np.ndarray:
        return self.loc
    
    @property
    def variance(self) -> np.ndarray:
        return 2 * self.scale ** 2
    
    def entropy(self) -> np.ndarray:
        """Entropy of Laplace distribution."""
        return 1 + np.log(2 * self.scale)


class Cauchy(Distribution):
    """
    Cauchy distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : float or array_like
        Location parameter
    scale : float or array_like
        Scale parameter
    
    Examples
    --------
    >>> dist = Cauchy(loc=0.0, scale=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        loc: Union[float, np.ndarray],
        scale: Union[float, np.ndarray]
    ):
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Cauchy distribution."""
        shape = sample_shape + self.loc.shape
        u = np.random.uniform(0, 1, shape)
        return (self.loc + self.scale * np.tan(np.pi * (u - 0.5))).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        return (-np.log(np.pi * self.scale) -
                np.log(1 + ((value - self.loc) / self.scale) ** 2))
    
    @property
    def mean(self) -> np.ndarray:
        return np.full_like(self.loc, np.nan)  # undefined
    
    @property
    def variance(self) -> np.ndarray:
        return np.full_like(self.scale, np.nan)  # undefined
    
    def entropy(self) -> np.ndarray:
        """Entropy of Cauchy distribution."""
        return np.log(4 * np.pi * self.scale)


class StudentT(Distribution):
    """
    Student's t-distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    df : float or array_like
        Degrees of freedom
    loc : float or array_like, default=0
        Location parameter
    scale : float or array_like, default=1
        Scale parameter
    
    Examples
    --------
    >>> dist = StudentT(df=5.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        df: Union[float, np.ndarray],
        loc: Union[float, np.ndarray] = 0.0,
        scale: Union[float, np.ndarray] = 1.0
    ):
        self.df = np.asarray(df, dtype=np.float32)
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Student's t distribution."""
        shape = sample_shape + self.df.shape
        t_samples = np.random.standard_t(self.df, shape)
        return (self.loc + self.scale * t_samples).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        y = (value - self.loc) / self.scale
        df = self.df
        return (special.gammaln((df + 1) / 2) - special.gammaln(df / 2) -
                0.5 * np.log(df * np.pi) - np.log(self.scale) -
                ((df + 1) / 2) * np.log(1 + y ** 2 / df))
    
    @property
    def mean(self) -> np.ndarray:
        return np.where(self.df > 1, self.loc, np.nan)
    
    @property
    def variance(self) -> np.ndarray:
        v = np.where(self.df > 2, self.df / (self.df - 2), np.inf)
        v = np.where(self.df <= 1, np.nan, v)
        return self.scale ** 2 * v


class Chi2(Distribution):
    """
    Chi-squared distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    df : float or array_like
        Degrees of freedom
    
    Examples
    --------
    >>> dist = Chi2(df=5)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(self, df: Union[float, np.ndarray]):
        self.df = np.asarray(df, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from chi-squared distribution."""
        shape = sample_shape + self.df.shape
        return np.random.chisquare(self.df, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        k = self.df / 2
        return ((k - 1) * np.log(value) - value / 2 - k * np.log(2) - special.gammaln(k))
    
    @property
    def mean(self) -> np.ndarray:
        return self.df
    
    @property
    def variance(self) -> np.ndarray:
        return 2 * self.df


class LogNormal(Distribution):
    """
    Log-Normal distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : float or array_like
        Mean of log of distribution
    scale : float or array_like
        Standard deviation of log of distribution
    
    Examples
    --------
    >>> dist = LogNormal(loc=0.0, scale=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        loc: Union[float, np.ndarray],
        scale: Union[float, np.ndarray]
    ):
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from log-normal distribution."""
        shape = sample_shape + self.loc.shape
        return np.random.lognormal(self.loc, self.scale, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        return (-np.log(value * self.scale * np.sqrt(2 * np.pi)) -
                (np.log(value) - self.loc) ** 2 / (2 * self.scale ** 2))
    
    @property
    def mean(self) -> np.ndarray:
        return np.exp(self.loc + self.scale ** 2 / 2)
    
    @property
    def variance(self) -> np.ndarray:
        return (np.exp(self.scale ** 2) - 1) * np.exp(2 * self.loc + self.scale ** 2)


class Gumbel(Distribution):
    """
    Gumbel distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : float or array_like
        Location parameter
    scale : float or array_like
        Scale parameter
    
    Examples
    --------
    >>> dist = Gumbel(loc=0.0, scale=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        loc: Union[float, np.ndarray],
        scale: Union[float, np.ndarray]
    ):
        self.loc = np.asarray(loc, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Gumbel distribution."""
        shape = sample_shape + self.loc.shape
        return np.random.gumbel(self.loc, self.scale, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        z = (value - self.loc) / self.scale
        return -z - np.exp(-z) - np.log(self.scale)
    
    @property
    def mean(self) -> np.ndarray:
        return self.loc + self.scale * np.euler_gamma
    
    @property
    def variance(self) -> np.ndarray:
        return (np.pi ** 2 / 6) * self.scale ** 2


class Weibull(Distribution):
    """
    Weibull distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    scale : float or array_like
        Scale parameter (lambda)
    concentration : float or array_like
        Concentration/shape parameter (k)
    
    Examples
    --------
    >>> dist = Weibull(scale=1.0, concentration=1.5)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        scale: Union[float, np.ndarray],
        concentration: Union[float, np.ndarray]
    ):
        self.scale = np.asarray(scale, dtype=np.float32)
        self.concentration = np.asarray(concentration, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Weibull distribution."""
        shape = sample_shape + self.scale.shape
        return (self.scale *
                np.random.weibull(self.concentration, shape)).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        k, lam = self.concentration, self.scale
        return (np.log(k / lam) + (k - 1) * np.log(value / lam) -
                (value / lam) ** k)
    
    @property
    def mean(self) -> np.ndarray:
        from scipy import special
        return self.scale * special.gamma(1 + 1 / self.concentration)
    
    @property
    def variance(self) -> np.ndarray:
        from scipy import special
        g1 = special.gamma(1 + 1 / self.concentration)
        g2 = special.gamma(1 + 2 / self.concentration)
        return self.scale ** 2 * (g2 - g1 ** 2)


class MultivariateNormal(Distribution):
    """
    Multivariate Normal distribution.
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : array_like
        Mean vector
    covariance_matrix : array_like, optional
        Covariance matrix
    precision_matrix : array_like, optional
        Precision matrix (inverse covariance)
    scale_tril : array_like, optional
        Lower triangular Cholesky factor
    
    Examples
    --------
    >>> dist = MultivariateNormal(loc=[0, 0], covariance_matrix=[[1, 0], [0, 1]])
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        loc: np.ndarray,
        covariance_matrix: Optional[np.ndarray] = None,
        precision_matrix: Optional[np.ndarray] = None,
        scale_tril: Optional[np.ndarray] = None
    ):
        self.loc = np.asarray(loc, dtype=np.float32)
        
        num_specified = sum(x is not None for x in [covariance_matrix, precision_matrix, scale_tril])
        if num_specified != 1:
            raise ValueError("Exactly one of covariance_matrix, precision_matrix, or scale_tril must be specified")
        
        if covariance_matrix is not None:
            self.covariance_matrix = np.asarray(covariance_matrix, dtype=np.float32)
            self.scale_tril = np.linalg.cholesky(self.covariance_matrix)
        elif precision_matrix is not None:
            precision = np.asarray(precision_matrix, dtype=np.float32)
            self.covariance_matrix = np.linalg.inv(precision)
            self.scale_tril = np.linalg.cholesky(self.covariance_matrix)
        else:
            self.scale_tril = np.asarray(scale_tril, dtype=np.float32)
            self.covariance_matrix = self.scale_tril @ self.scale_tril.T
    
    @property
    def dimension(self) -> int:
        return len(self.loc)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from multivariate normal distribution."""
        shape = sample_shape
        z = np.random.standard_normal(shape + (self.dimension,)).astype(np.float32)
        return self.loc + z @ self.scale_tril.T
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        diff = value - self.loc
        precision = np.linalg.inv(self.covariance_matrix)
        log_det = np.log(np.linalg.det(self.covariance_matrix))
        mahal = np.sum(diff @ precision * diff, axis=-1)
        return -0.5 * (self.dimension * np.log(2 * np.pi) + log_det + mahal)
    
    @property
    def mean(self) -> np.ndarray:
        return self.loc
    
    @property
    def variance(self) -> np.ndarray:
        return np.diag(self.covariance_matrix)
    
    def entropy(self) -> np.ndarray:
        """Entropy of multivariate normal distribution."""
        log_det = np.log(np.linalg.det(self.covariance_matrix))
        return 0.5 * (self.dimension * (1 + np.log(2 * np.pi)) + log_det)


class VonMises(Distribution):
    """
    Von Mises distribution (circular/directional).
    
    Neurova implementation.
    
    Parameters
    ----------
    loc : float or array_like
        Mean direction (radians)
    concentration : float or array_like
        Concentration parameter (kappa)
    
    Examples
    --------
    >>> dist = VonMises(loc=0.0, concentration=1.0)
    >>> samples = dist.sample((100,))
    """
    
    def __init__(
        self,
        loc: Union[float, np.ndarray],
        concentration: Union[float, np.ndarray]
    ):
        self.loc = np.asarray(loc, dtype=np.float32)
        self.concentration = np.asarray(concentration, dtype=np.float32)
    
    def sample(self, sample_shape: Tuple[int, ...] = ()) -> np.ndarray:
        """Sample from Von Mises distribution."""
        shape = sample_shape + self.loc.shape
        return np.random.vonmises(self.loc, self.concentration, shape).astype(np.float32)
    
    def log_prob(self, value: np.ndarray) -> np.ndarray:
        """Compute log probability."""
        from scipy import special
        return (self.concentration * np.cos(value - self.loc) -
                np.log(2 * np.pi * special.i0(self.concentration)))
    
    @property
    def mean(self) -> np.ndarray:
        return self.loc
    
    @property
    def variance(self) -> np.ndarray:
        from scipy import special
        return 1 - special.i1(self.concentration) / special.i0(self.concentration)


# Helper functions
def kl_divergence(p: Distribution, q: Distribution) -> np.ndarray:
    """
    Compute KL divergence between two distributions.
    
    Parameters
    ----------
    p : Distribution
        First distribution
    q : Distribution
        Second distribution
    
    Returns
    -------
    np.ndarray
        KL(p || q)
    """
    # For special cases, compute analytically
    if isinstance(p, Normal) and isinstance(q, Normal):
        var_ratio = p.variance / q.variance
        t1 = ((p.loc - q.loc) ** 2) / q.variance
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))
    
    # Monte Carlo estimate for general case
    samples = p.sample((1000,))
    return np.mean(p.log_prob(samples) - q.log_prob(samples))


# Export all distributions
__all__ = [
    'Distribution',
    'Normal',
    'Uniform',
    'Bernoulli',
    'Categorical',
    'Exponential',
    'Gamma',
    'Beta',
    'Poisson',
    'Binomial',
    'Dirichlet',
    'Laplace',
    'Cauchy',
    'StudentT',
    'Chi2',
    'LogNormal',
    'Gumbel',
    'Weibull',
    'MultivariateNormal',
    'VonMises',
    'kl_divergence',
]
