# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Special Functions Module.

Neurova implementation of special mathematical functions
commonly used in scientific computing and deep learning.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union
from scipy import special as scipy_special


# Gamma and Related Functions

def gamma(x: np.ndarray) -> np.ndarray:
    """
    Compute the Gamma function Γ(x).
    
    The Gamma function is defined as:
    Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Gamma function values
    """
    return scipy_special.gamma(x)


def lgamma(x: np.ndarray) -> np.ndarray:
    """
    Compute the log of the absolute value of Gamma function.
    
    Args:
        x: Input tensor
        
    Returns:
        log|Γ(x)| values
    """
    return scipy_special.gammaln(np.abs(x))


def digamma(x: np.ndarray) -> np.ndarray:
    """
    Compute the digamma function ψ(x) = d/dx ln(Γ(x)).
    
    Args:
        x: Input tensor
        
    Returns:
        Digamma function values
    """
    return scipy_special.digamma(x)


def polygamma(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute the n-th derivative of the digamma function.
    
    Args:
        n: Order of derivative
        x: Input tensor
        
    Returns:
        Polygamma function values
    """
    return scipy_special.polygamma(n, x)


def multigammaln(a: np.ndarray, d: int) -> np.ndarray:
    """
    Compute log of multivariate gamma function.
    
    Γ_d(a) = π^(d(d-1)/4) ∏_{j=1}^{d} Γ(a + (1-j)/2)
    
    Args:
        a: Input tensor
        d: Dimension
        
    Returns:
        Log multivariate gamma values
    """
    return scipy_special.multigammaln(a, d)


def gammaln(x: np.ndarray) -> np.ndarray:
    """
    Compute log of Gamma function.
    
    Args:
        x: Input tensor (must be positive)
        
    Returns:
        ln(Γ(x)) values
    """
    return scipy_special.gammaln(x)


def gammainc(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute regularized lower incomplete gamma function.
    
    P(a, x) = (1/Γ(a)) ∫₀^x t^(a-1) e^(-t) dt
    
    Args:
        a: Shape parameter
        x: Upper limit
        
    Returns:
        Incomplete gamma values
    """
    return scipy_special.gammainc(a, x)


def gammaincc(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute regularized upper incomplete gamma function.
    
    Q(a, x) = 1 - P(a, x)
    
    Args:
        a: Shape parameter
        x: Lower limit
        
    Returns:
        Complementary incomplete gamma values
    """
    return scipy_special.gammaincc(a, x)


# Beta Function

def beta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the Beta function B(a, b).
    
    B(a, b) = Γ(a)Γ(b) / Γ(a+b)
    
    Args:
        a: First parameter
        b: Second parameter
        
    Returns:
        Beta function values
    """
    return scipy_special.beta(a, b)


def betaln(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute log of Beta function.
    
    Args:
        a: First parameter
        b: Second parameter
        
    Returns:
        ln(B(a, b)) values
    """
    return scipy_special.betaln(a, b)


def betainc(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute regularized incomplete beta function.
    
    I_x(a, b) = B(x; a, b) / B(a, b)
    
    Args:
        a: First parameter
        b: Second parameter
        x: Upper limit
        
    Returns:
        Incomplete beta values
    """
    return scipy_special.betainc(a, b, x)


def betaincinv(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute inverse of regularized incomplete beta function.
    
    Args:
        a: First parameter
        b: Second parameter
        y: Probability value
        
    Returns:
        x such that I_x(a, b) = y
    """
    return scipy_special.betaincinv(a, b, y)


# Error Function

def erf(x: np.ndarray) -> np.ndarray:
    """
    Compute the error function.
    
    erf(x) = (2/√π) ∫₀^x e^(-t²) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Error function values
    """
    return scipy_special.erf(x)


def erfc(x: np.ndarray) -> np.ndarray:
    """
    Compute the complementary error function.
    
    erfc(x) = 1 - erf(x)
    
    Args:
        x: Input tensor
        
    Returns:
        Complementary error function values
    """
    return scipy_special.erfc(x)


def erfinv(x: np.ndarray) -> np.ndarray:
    """
    Compute the inverse error function.
    
    Args:
        x: Input tensor (must be in [-1, 1])
        
    Returns:
        Inverse error function values
    """
    return scipy_special.erfinv(x)


def erfcinv(x: np.ndarray) -> np.ndarray:
    """
    Compute the inverse complementary error function.
    
    Args:
        x: Input tensor (must be in [0, 2])
        
    Returns:
        Inverse complementary error function values
    """
    return scipy_special.erfcinv(x)


def erfi(x: np.ndarray) -> np.ndarray:
    """
    Compute the imaginary error function.
    
    erfi(x) = -i * erf(i*x)
    
    Args:
        x: Input tensor
        
    Returns:
        Imaginary error function values
    """
    return np.real(-1j * scipy_special.erf(1j * x))


def dawsn(x: np.ndarray) -> np.ndarray:
    """
    Compute Dawson's integral.
    
    F(x) = exp(-x²) ∫₀^x exp(t²) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Dawson's integral values
    """
    return scipy_special.dawsn(x)


# Bessel Functions

def bessel_j0(x: np.ndarray) -> np.ndarray:
    """Compute Bessel function of first kind, order 0."""
    return scipy_special.j0(x)


def bessel_j1(x: np.ndarray) -> np.ndarray:
    """Compute Bessel function of first kind, order 1."""
    return scipy_special.j1(x)


def bessel_jn(n: int, x: np.ndarray) -> np.ndarray:
    """Compute Bessel function of first kind, order n."""
    return scipy_special.jn(n, x)


def bessel_y0(x: np.ndarray) -> np.ndarray:
    """Compute Bessel function of second kind, order 0."""
    return scipy_special.y0(x)


def bessel_y1(x: np.ndarray) -> np.ndarray:
    """Compute Bessel function of second kind, order 1."""
    return scipy_special.y1(x)


def bessel_yn(n: int, x: np.ndarray) -> np.ndarray:
    """Compute Bessel function of second kind, order n."""
    return scipy_special.yn(n, x)


def bessel_i0(x: np.ndarray) -> np.ndarray:
    """Compute modified Bessel function of first kind, order 0."""
    return scipy_special.i0(x)


def bessel_i0e(x: np.ndarray) -> np.ndarray:
    """Compute exponentially scaled modified Bessel function I₀."""
    return scipy_special.i0e(x)


def bessel_i1(x: np.ndarray) -> np.ndarray:
    """Compute modified Bessel function of first kind, order 1."""
    return scipy_special.i1(x)


def bessel_i1e(x: np.ndarray) -> np.ndarray:
    """Compute exponentially scaled modified Bessel function I₁."""
    return scipy_special.i1e(x)


def bessel_iv(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute modified Bessel function of first kind, order v."""
    return scipy_special.iv(v, x)


def bessel_k0(x: np.ndarray) -> np.ndarray:
    """Compute modified Bessel function of second kind, order 0."""
    return scipy_special.k0(x)


def bessel_k0e(x: np.ndarray) -> np.ndarray:
    """Compute exponentially scaled modified Bessel function K₀."""
    return scipy_special.k0e(x)


def bessel_k1(x: np.ndarray) -> np.ndarray:
    """Compute modified Bessel function of second kind, order 1."""
    return scipy_special.k1(x)


def bessel_k1e(x: np.ndarray) -> np.ndarray:
    """Compute exponentially scaled modified Bessel function K₁."""
    return scipy_special.k1e(x)


def bessel_kv(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute modified Bessel function of second kind, order v."""
    return scipy_special.kv(v, x)


def spherical_bessel_j(n: int, x: np.ndarray) -> np.ndarray:
    """Compute spherical Bessel function of first kind."""
    return scipy_special.spherical_jn(n, x)


def spherical_bessel_y(n: int, x: np.ndarray) -> np.ndarray:
    """Compute spherical Bessel function of second kind."""
    return scipy_special.spherical_yn(n, x)


# Airy Functions

def airy(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Airy functions Ai(x), Ai'(x), Bi(x), Bi'(x).
    
    Args:
        x: Input tensor
        
    Returns:
        Tuple of (Ai, Aip, Bi, Bip)
    """
    return scipy_special.airy(x)


def airy_ai(x: np.ndarray) -> np.ndarray:
    """Compute Airy function Ai(x)."""
    return scipy_special.airy(x)[0]


def airy_bi(x: np.ndarray) -> np.ndarray:
    """Compute Airy function Bi(x)."""
    return scipy_special.airy(x)[2]


# Elliptic Functions

def elliptic_k(m: np.ndarray) -> np.ndarray:
    """
    Compute complete elliptic integral of first kind K(m).
    
    K(m) = ∫₀^(π/2) dθ / √(1 - m sin²θ)
    
    Args:
        m: Parameter (0 ≤ m ≤ 1)
        
    Returns:
        Complete elliptic integral values
    """
    return scipy_special.ellipk(m)


def elliptic_e(m: np.ndarray) -> np.ndarray:
    """
    Compute complete elliptic integral of second kind E(m).
    
    E(m) = ∫₀^(π/2) √(1 - m sin²θ) dθ
    
    Args:
        m: Parameter (0 ≤ m ≤ 1)
        
    Returns:
        Complete elliptic integral values
    """
    return scipy_special.ellipe(m)


def elliptic_f(phi: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Compute incomplete elliptic integral of first kind F(φ, m).
    
    Args:
        phi: Amplitude angle
        m: Parameter
        
    Returns:
        Incomplete elliptic integral values
    """
    return scipy_special.ellipkinc(phi, m)


def elliptic_pi(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Compute complete elliptic integral of third kind Π(n, m).
    
    Args:
        n: Characteristic
        m: Parameter
        
    Returns:
        Complete elliptic integral values
    """
    # Using numerical integration as scipy doesn't have direct Π
    from scipy import integrate
    
    def integrand(theta, n, m):
        return 1 / ((1 - n * np.sin(theta)**2) * np.sqrt(1 - m * np.sin(theta)**2))
    
    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    result = np.zeros_like(n, dtype=float)
    
    for i in range(len(n.flat)):
        result.flat[i], _ = integrate.quad(integrand, 0, np.pi/2, args=(n.flat[i], m.flat[i]))
    
    return result.reshape(n.shape)


# Legendre Functions

def legendre_p(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Legendre polynomial P_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Legendre polynomial values
    """
    return scipy_special.eval_legendre(n, x)


def legendre_q(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Legendre function of second kind Q_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Legendre function values
    """
    return scipy_special.lqn(n, x)[0][-1]


def associated_legendre_p(n: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    Compute associated Legendre function P_n^m(x).
    
    Args:
        n: Degree
        m: Order
        x: Input tensor
        
    Returns:
        Associated Legendre function values
    """
    return scipy_special.lpmv(m, n, x)


def spherical_harmonic(m: int, n: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Compute spherical harmonic Y_n^m(θ, φ).
    
    Args:
        m: Order (-n ≤ m ≤ n)
        n: Degree
        theta: Polar angle
        phi: Azimuthal angle
        
    Returns:
        Complex spherical harmonic values
    """
    return scipy_special.sph_harm(m, n, phi, theta)


# Chebyshev Polynomials

def chebyshev_t(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev polynomial of first kind T_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Chebyshev polynomial values
    """
    return scipy_special.eval_chebyt(n, x)


def chebyshev_u(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev polynomial of second kind U_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Chebyshev polynomial values
    """
    return scipy_special.eval_chebyu(n, x)


# Other Orthogonal Polynomials

def hermite_h(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute physicists' Hermite polynomial H_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Hermite polynomial values
    """
    return scipy_special.eval_hermite(n, x)


def hermite_he(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute probabilists' Hermite polynomial He_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Hermite polynomial values
    """
    return scipy_special.eval_hermitenorm(n, x)


def laguerre_l(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute Laguerre polynomial L_n(x).
    
    Args:
        n: Degree
        x: Input tensor
        
    Returns:
        Laguerre polynomial values
    """
    return scipy_special.eval_laguerre(n, x)


def generalized_laguerre(n: int, alpha: float, x: np.ndarray) -> np.ndarray:
    """
    Compute generalized Laguerre polynomial L_n^α(x).
    
    Args:
        n: Degree
        alpha: Parameter
        x: Input tensor
        
    Returns:
        Generalized Laguerre polynomial values
    """
    return scipy_special.eval_genlaguerre(n, alpha, x)


def jacobi_p(n: int, alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    """
    Compute Jacobi polynomial P_n^(α,β)(x).
    
    Args:
        n: Degree
        alpha: First parameter
        beta: Second parameter
        x: Input tensor
        
    Returns:
        Jacobi polynomial values
    """
    return scipy_special.eval_jacobi(n, alpha, beta, x)


# Hypergeometric Functions

def hyp2f1(a: np.ndarray, b: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute Gauss hypergeometric function 2F1(a, b; c; z).
    
    Args:
        a, b: Numerator parameters
        c: Denominator parameter
        z: Argument
        
    Returns:
        Hypergeometric function values
    """
    return scipy_special.hyp2f1(a, b, c, z)


def hyp1f1(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute confluent hypergeometric function 1F1(a; b; z).
    
    Also known as Kummer's function M(a, b, z).
    
    Args:
        a: Numerator parameter
        b: Denominator parameter
        z: Argument
        
    Returns:
        Confluent hypergeometric values
    """
    return scipy_special.hyp1f1(a, b, z)


def hyp0f1(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute confluent hypergeometric limit function 0F1(v; z).
    
    Args:
        v: Parameter
        z: Argument
        
    Returns:
        Hypergeometric limit function values
    """
    return scipy_special.hyp0f1(v, z)


# Zeta and Related Functions

def zeta(s: np.ndarray, q: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Hurwitz zeta function ζ(s, q) or Riemann zeta function ζ(s).
    
    Args:
        s: Exponent
        q: Offset (if None, computes Riemann zeta)
        
    Returns:
        Zeta function values
    """
    if q is None:
        return scipy_special.zeta(s, 1)
    return scipy_special.zeta(s, q)


def riemann_zeta(s: np.ndarray) -> np.ndarray:
    """
    Compute Riemann zeta function ζ(s).
    
    Args:
        s: Input tensor
        
    Returns:
        Riemann zeta values
    """
    return scipy_special.zeta(s, 1)


# Exponential and Logarithmic Integrals

def expi(x: np.ndarray) -> np.ndarray:
    """
    Compute exponential integral Ei(x).
    
    Ei(x) = ∫_{-∞}^x (e^t / t) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Exponential integral values
    """
    return scipy_special.expi(x)


def exp1(x: np.ndarray) -> np.ndarray:
    """
    Compute exponential integral E₁(x).
    
    E₁(x) = ∫_x^∞ (e^{-t} / t) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Exponential integral values
    """
    return scipy_special.exp1(x)


def expn(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute generalized exponential integral E_n(x).
    
    Args:
        n: Order
        x: Input tensor
        
    Returns:
        Generalized exponential integral values
    """
    return scipy_special.expn(n, x)


def spence(x: np.ndarray) -> np.ndarray:
    """
    Compute Spence's function (dilogarithm).
    
    Li₂(x) = -∫_0^x (ln(1-t)/t) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Dilogarithm values
    """
    return scipy_special.spence(x)


# Fresnel Integrals

def fresnel(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fresnel integrals S(x) and C(x).
    
    S(x) = ∫_0^x sin(πt²/2) dt
    C(x) = ∫_0^x cos(πt²/2) dt
    
    Args:
        x: Input tensor
        
    Returns:
        Tuple of (S, C)
    """
    return scipy_special.fresnel(x)


# Other Special Functions

def sinc(x: np.ndarray) -> np.ndarray:
    """
    Compute sinc function sin(πx)/(πx).
    
    Args:
        x: Input tensor
        
    Returns:
        Sinc function values
    """
    return np.sinc(x)


def log1p(x: np.ndarray) -> np.ndarray:
    """
    Compute log(1 + x) accurately for small x.
    
    Args:
        x: Input tensor
        
    Returns:
        log(1 + x) values
    """
    return np.log1p(x)


def expm1(x: np.ndarray) -> np.ndarray:
    """
    Compute exp(x) - 1 accurately for small x.
    
    Args:
        x: Input tensor
        
    Returns:
        exp(x) - 1 values
    """
    return np.expm1(x)


def log_ndtr(x: np.ndarray) -> np.ndarray:
    """
    Compute log of standard normal CDF.
    
    Args:
        x: Input tensor
        
    Returns:
        log(Φ(x)) values
    """
    return scipy_special.log_ndtr(x)


def ndtr(x: np.ndarray) -> np.ndarray:
    """
    Compute standard normal CDF.
    
    Args:
        x: Input tensor
        
    Returns:
        Φ(x) values
    """
    return scipy_special.ndtr(x)


def ndtri(p: np.ndarray) -> np.ndarray:
    """
    Compute inverse of standard normal CDF.
    
    Args:
        p: Probability values
        
    Returns:
        Φ⁻¹(p) values
    """
    return scipy_special.ndtri(p)


def xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute x * log(y), returning 0 where x = 0.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        x * log(y), with proper handling of x = 0
    """
    return scipy_special.xlogy(x, y)


def xlog1py(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute x * log(1 + y), returning 0 where x = 0.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        x * log(1 + y), with proper handling of x = 0
    """
    return scipy_special.xlog1py(x, y)


def entr(x: np.ndarray) -> np.ndarray:
    """
    Compute elementwise entropy -x * log(x).
    
    Args:
        x: Input tensor (positive values)
        
    Returns:
        Entropy values
    """
    return scipy_special.entr(x)


def rel_entr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute elementwise relative entropy x * log(x/y).
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Relative entropy values
    """
    return scipy_special.rel_entr(x, y)


def comb(n: np.ndarray, k: np.ndarray, exact: bool = False) -> np.ndarray:
    """
    Compute binomial coefficient C(n, k).
    
    Args:
        n: Upper index
        k: Lower index
        exact: Use exact integer arithmetic
        
    Returns:
        Binomial coefficient values
    """
    return scipy_special.comb(n, k, exact=exact)


def perm(n: np.ndarray, k: np.ndarray, exact: bool = False) -> np.ndarray:
    """
    Compute permutation coefficient P(n, k) = n! / (n-k)!.
    
    Args:
        n: Upper index
        k: Lower index
        exact: Use exact integer arithmetic
        
    Returns:
        Permutation coefficient values
    """
    return scipy_special.perm(n, k, exact=exact)


def factorial(n: np.ndarray, exact: bool = False) -> np.ndarray:
    """
    Compute factorial n!.
    
    Args:
        n: Input tensor (non-negative integers)
        exact: Use exact integer arithmetic
        
    Returns:
        Factorial values
    """
    return scipy_special.factorial(n, exact=exact)


def factorial2(n: np.ndarray, exact: bool = False) -> np.ndarray:
    """
    Compute double factorial n!!.
    
    Args:
        n: Input tensor (non-negative integers)
        exact: Use exact integer arithmetic
        
    Returns:
        Double factorial values
    """
    return scipy_special.factorial2(n, exact=exact)


# Export all functions
__all__ = [
    # Gamma functions
    'gamma', 'lgamma', 'digamma', 'polygamma', 'multigammaln', 'gammaln',
    'gammainc', 'gammaincc',
    # Beta functions
    'beta', 'betaln', 'betainc', 'betaincinv',
    # Error functions
    'erf', 'erfc', 'erfinv', 'erfcinv', 'erfi', 'dawsn',
    # Bessel functions
    'bessel_j0', 'bessel_j1', 'bessel_jn', 'bessel_y0', 'bessel_y1', 'bessel_yn',
    'bessel_i0', 'bessel_i0e', 'bessel_i1', 'bessel_i1e', 'bessel_iv',
    'bessel_k0', 'bessel_k0e', 'bessel_k1', 'bessel_k1e', 'bessel_kv',
    'spherical_bessel_j', 'spherical_bessel_y',
    # Airy functions
    'airy', 'airy_ai', 'airy_bi',
    # Elliptic functions
    'elliptic_k', 'elliptic_e', 'elliptic_f', 'elliptic_pi',
    # Legendre functions
    'legendre_p', 'legendre_q', 'associated_legendre_p', 'spherical_harmonic',
    # Chebyshev polynomials
    'chebyshev_t', 'chebyshev_u',
    # Other polynomials
    'hermite_h', 'hermite_he', 'laguerre_l', 'generalized_laguerre', 'jacobi_p',
    # Hypergeometric functions
    'hyp2f1', 'hyp1f1', 'hyp0f1',
    # Zeta functions
    'zeta', 'riemann_zeta',
    # Exponential integrals
    'expi', 'exp1', 'expn', 'spence',
    # Fresnel integrals
    'fresnel',
    # Other functions
    'sinc', 'log1p', 'expm1', 'log_ndtr', 'ndtr', 'ndtri',
    'xlogy', 'xlog1py', 'entr', 'rel_entr',
    'comb', 'perm', 'factorial', 'factorial2',
]
