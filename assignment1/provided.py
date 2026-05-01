"""Staff-provided helpers for NTT parameter generation and utilities.

This module provides utilities you may find helpful, but you're not required
to use them. The main entry points are:

- ``generate_ntt_modulus(N, bit_length)``: Find a prime q suitable for NTT
- ``negacyclic_psi(N, q)``: Find a primitive 2N-th root of unity mod q
- ``negacyclic_psi_from_max(psi_max, N_max, N, q)``: Derive psi for size N
- ``broadcast_to_axis(arr, like, axis=0)``: Broadcast a vector along an axis

Examples:
    >>> import provided
    >>> N = 1024
    >>> q = provided.generate_ntt_modulus(N, bit_length=31)
    >>> psi = provided.negacyclic_psi(N, q)
    >>> pow(psi, N, q) == q - 1  # psi^N = -1 (mod q)
    True
"""

from __future__ import annotations

from functools import lru_cache

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from sympy import isprime


# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------


def broadcast_to_axis(arr, like, axis: int = 0) -> jnp.ndarray:
    """Reshape a 1D array to broadcast along a specific axis.

    Args:
        arr: Scalar, 1D array, or array with same rank as ``like``.
        like: Reference tensor whose shape determines the broadcast target.
        axis: Axis of ``like`` that ``arr`` should align with.

    Returns:
        Array reshaped to be broadcast-compatible with ``like``.

    Raises:
        ValueError: If arr is not scalar, 1D, or same rank as like.
        ValueError: If axis is out of bounds.
        ValueError: If arr length doesn't match like.shape[axis].

    Examples:
        >>> x = jnp.ones((4, 8, 16))
        >>> q = jnp.array([17, 31, 37, 41])
        >>> broadcast_to_axis(q, x, axis=0).shape
        (4, 1, 1)
    """
    arr = jnp.asarray(arr)
    if arr.ndim in (0, like.ndim):
        return arr
    if arr.ndim != 1:
        raise ValueError(
            f"Expected scalar, 1D, or rank-{like.ndim} array, "
            f"got rank {arr.ndim}"
        )

    rank = like.ndim
    if not -rank <= axis < rank:
        raise ValueError(f"axis {axis} out of range for rank {rank}")
    axis = axis % rank

    if arr.shape[0] != like.shape[axis] and like.shape[axis] != 1:
        raise ValueError(
            f"Length {arr.shape[0]} doesn't match like.shape[{axis}]="
            f"{like.shape[axis]}"
        )

    shape = [1] * rank
    shape[axis] = arr.shape[0]
    return arr.reshape(shape)



# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def precompute_tables(N, q, psi):
    """
    Precompute power tables for negacyclic NTT.

    Args:
        N: Transform size (must be power of two)
        q: Prime modulus
        psi: Primitive 2N-th root of unity

    Returns:
        tuple: (psi_powers, twiddles) as uint32 arrays

    Where:
        psi_powers[n] = ψ^n mod q (for negacyclic twist)
        twiddles[span:2*span] = stage twiddles for Stockham NTT
    """
    if N <= 0 or N & (N - 1) != 0:
        raise ValueError(f"N must be a positive power of two, got {N}")

    q, psi = int(q), int(psi)
    omega = pow(psi, 2, q)

    # ψ^n for negacyclic twist (use Python int to avoid uint32 overflow)
    psi_powers = np.empty(N, dtype=np.uint32)
    cur = 1
    for i in range(N):
        psi_powers[i] = cur
        cur = (cur * psi) % q

    # Stockham stage twiddles for cyclic NTT with ω = ψ²
    twiddles = np.ones(N, dtype=np.uint32)
    stages = N.bit_length() - 1

    for s in range(stages):
        span = 1 << s
        stride = N // (2 * span)
        step = pow(omega, stride, q)
        cur = 1
        for j in range(span):
            twiddles[span + j] = cur
            cur = (cur * step) % q

    return psi_powers, twiddles


# ---------------------------------------------------------------------------
# Primitive roots
# ---------------------------------------------------------------------------


def prime_factors(n: int) -> list[int]:
    """Return distinct prime factors of n in ascending order.

    Examples:
        >>> prime_factors(60)
        [2, 3, 5]
        >>> prime_factors(17)
        [17]
    """
    factors: list[int] = []
    d = 2
    x = int(n)

    while d * d <= x:
        if x % d == 0:
            factors.append(d)
            while x % d == 0:
                x //= d
        d = 3 if d == 2 else d + 2

    if x > 1:
        factors.append(x)
    return factors


def find_generator(modulus: int) -> int:
    """Find a generator (primitive root) of the multiplicative group (Z/qZ)*.

    A generator g has order phi(q) = q-1, meaning g^k cycles through all
    nonzero residues before returning to 1.

    Args:
        modulus: A prime number.

    Returns:
        Smallest generator g >= 2.

    Raises:
        ValueError: If no generator is found (shouldn't happen for primes).
    """
    if modulus == 2:
        return 1

    phi = modulus - 1
    factors = prime_factors(phi)

    for g in range(2, modulus):
        if all(pow(g, phi // f, modulus) != 1 for f in factors):
            return g
    raise ValueError(f"No generator found for modulus {modulus}")


def find_primitive_root(order: int, modulus: int) -> int:
    """Find a primitive k-th root of unity modulo q.

    A primitive k-th root r satisfies r^k = 1 and r^j != 1 for 0 < j < k.

    Args:
        order: The desired root order k.
        modulus: A prime q where (q - 1) is divisible by k.

    Returns:
        A primitive k-th root of unity mod q.

    Raises:
        ValueError: If order does not divide (modulus - 1).

    Examples:
        >>> r = find_primitive_root(8, 17)  # 8 divides 16
        >>> pow(r, 8, 17)
        1
        >>> pow(r, 4, 17) != 1
        True
    """
    phi = modulus - 1
    if phi % order != 0:
        raise ValueError(
            f"Order {order} does not divide modulus-1 ({phi})"
        )
    g = find_generator(modulus)
    return pow(g, phi // order, modulus)


# ---------------------------------------------------------------------------
# NTT modulus and root generation
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def generate_ntt_modulus(N: int, bit_length: int = 31) -> int:
    """Find a prime q suitable for negacyclic NTT of size N.

    For negacyclic NTT, we need a primitive 2N-th root of unity mod q.
    This exists when q = 1 (mod 2N). This function searches downward
    from 2^bit_length to find such a prime.

    Args:
        N: Transform size (must be positive).
        bit_length: Upper bound on prime size: q < 2^bit_length.

    Returns:
        A prime q with (q - 1) % (2N) == 0 and q < 2^bit_length.

    Raises:
        ValueError: If N or bit_length is not positive.
        RuntimeError: If no suitable prime is found.

    Examples:
        >>> q = generate_ntt_modulus(1024, bit_length=31)
        >>> (q - 1) % 2048 == 0
        True
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    if bit_length <= 0:
        raise ValueError(f"bit_length must be positive, got {bit_length}")

    step = 2 * int(N)

    # Start from largest candidate below 2^bit_length that is 1 (mod 2N)
    limit = (1 << int(bit_length)) - 1
    k = (limit - 1) // step
    candidate = k * step + 1

    while candidate >= 2:
        if isprime(candidate):
            return int(candidate)
        candidate -= step

    raise RuntimeError(
        f"No prime q = 1 (mod {step}) found below 2^{bit_length}"
    )


def negacyclic_psi(N: int, q: int) -> int:
    """Find a primitive 2N-th root of unity mod q for negacyclic NTT.

    The returned psi satisfies psi^(2N) = 1 and psi^N = -1 (mod q).

    Args:
        N: Transform size.
        q: Prime modulus with (q - 1) divisible by 2N.

    Returns:
        A primitive 2N-th root of unity mod q.

    Raises:
        ValueError: If the computed root doesn't satisfy psi^N = -1.

    Examples:
        >>> q = generate_ntt_modulus(4, bit_length=8)
        >>> psi = negacyclic_psi(4, q)
        >>> pow(psi, 4, q) == q - 1  # psi^N = -1
        True
    """
    psi = find_primitive_root(2 * int(N), int(q))
    if pow(psi, int(N), int(q)) != int(q) - 1:
        raise ValueError("psi^N != -1 (mod q), invalid root")
    return int(psi)


def negacyclic_psi_from_max(psi_max: int, N_max: int, N: int, q: int) -> int:
    """Derive a 2N-th root from a 2*N_max-th root.

    When N divides N_max, we can compute psi_N = psi_max^(N_max/N).
    This is useful when benchmarking multiple sizes with a shared modulus.

    Args:
        psi_max: Primitive 2*N_max-th root of unity mod q.
        N_max: The larger transform size.
        N: The smaller transform size (must divide N_max).
        q: Prime modulus.

    Returns:
        A primitive 2N-th root of unity mod q.

    Raises:
        ValueError: If N does not divide N_max.

    Examples:
        >>> q = generate_ntt_modulus(1024, bit_length=31)
        >>> psi_1024 = negacyclic_psi(1024, q)
        >>> psi_256 = negacyclic_psi_from_max(psi_1024, 1024, 256, q)
        >>> pow(psi_256, 256, q) == q - 1
        True
    """
    if N_max % N != 0:
        raise ValueError(f"N={N} must divide N_max={N_max}")
    exp = N_max // N
    return int(pow(int(psi_max), int(exp), int(q)))
