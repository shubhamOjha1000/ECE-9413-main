"""
Reference negacyclic NTT implementation using SymPy.

Provides a trusted oracle for testing. Uses SymPy's cyclic NTT with the
standard twist trick: multiply x[n] by ψ^n, then apply cyclic NTT with ω = ψ².
"""

from __future__ import annotations

from functools import lru_cache

from sympy.discrete.transforms import ntt as sympy_ntt


@lru_cache(maxsize=None)
def psi_powers(psi, N, q):
    """Precompute [ψ⁰, ψ¹, ..., ψ^{N-1}] mod q."""
    return [pow(psi, i, q) for i in range(N)]


def negacyclic_ntt_oracle(x, *, q, psi):
    """
    Compute negacyclic NTT using SymPy as reference.

    Args:
        x: Input coefficients, length N, values in [0, q)
        q: Prime modulus
        psi: Primitive 2N-th root of unity (ψ^N ≡ -1 mod q)

    Returns:
        list[int]: NTT outputs, length N, values in [0, q)

    Example:
        >>> negacyclic_ntt_oracle([1, 2, 3, 4], q=17, psi=2)
        [6, 15, 9, 4]
    """
    x = [int(v) % q for v in x]
    N = len(x)

    # Twist: x'[n] = x[n] · ψ^n
    tw = psi_powers(psi, N, q)
    x_twisted = [(x[i] * tw[i]) % q for i in range(N)]

    # Cyclic NTT with ω = ψ²
    y = sympy_ntt(x_twisted, q)

    return [int(v) % q for v in y]