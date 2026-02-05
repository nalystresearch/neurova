# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Neurova accelerated backends."""

try:
    from . import cuda_native  # noqa: F401
    HAS_CUDA_NATIVE = True
except Exception:  # pragma: no cover - optional binding
    cuda_native = None
    HAS_CUDA_NATIVE = False

__all__ = ["cuda_native", "HAS_CUDA_NATIVE"]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.