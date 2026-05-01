"""
Public tests for the negacyclic NTT implementation.

Covers:
- Reference agreement across sizes and batches
- JIT and vmap consistency
- Linearity in the ring

Usage:
    pytest tests/test_ntt.py --logn 10 --batch 4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import provided
import student
from tests.reference import negacyclic_ntt_oracle

DEFAULT_SEED = 42


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ntt_params(logn):
    """Generate shared (N, q, psi, psi_powers, twiddles) for all tests."""
    N = 1 << logn
    q = provided.generate_ntt_modulus(N, bit_length=31)
    psi = provided.negacyclic_psi(N, q)
    psi_powers, twiddles = provided.precompute_tables(N, q, psi)
    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles = jnp.asarray(twiddles, dtype=jnp.uint32)
    return N, q, psi, psi_powers, twiddles


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def random_input(rng, q, shape):
    """
    Generate random uint32 array with values in [0, q).

    Args:
        rng: NumPy random generator
        q: Modulus (exclusive upper bound)
        shape: Output shape

    Returns:
        jnp.ndarray: Random values as uint32
    """
    x = rng.integers(0, q, size=shape, dtype=np.int64)
    return jnp.asarray(x, dtype=jnp.uint32)


def to_int64(x):
    """Convert JAX array to int64 NumPy array."""
    return np.asarray(x, dtype=np.int64)


def reference_ntt(x_np, q, psi):
    """
    Compute batched reference NTT using SymPy oracle.

    Args:
        x_np: Input array, shape (batch, N)
        q: Modulus
        psi: Primitive 2N-th root of unity

    Returns:
        np.ndarray: Reference outputs, shape (batch, N)
    """
    refs = [negacyclic_ntt_oracle(row.tolist(), q=q, psi=psi) for row in x_np]
    return np.asarray(refs, dtype=np.int64)


# -----------------------------------------------------------------------------
# Correctness Tests
# -----------------------------------------------------------------------------

def test_matches_reference(batch, ntt_params):
    """NTT output matches SymPy reference across sizes and batch shapes."""
    N, q, psi, psi_powers, twiddles = ntt_params
    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    rng = np.random.default_rng(DEFAULT_SEED)
    x = random_input(rng, q, shape=(batch, N))
    y = student.ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)

    assert y.shape == x.shape

    y_np = to_int64(y) % q
    ref = reference_ntt(to_int64(x), q, psi)
    np.testing.assert_array_equal(y_np, ref)


# -----------------------------------------------------------------------------
# JAX Compatibility Tests
# -----------------------------------------------------------------------------

def test_jit_matches_eager(batch, ntt_params):
    """JIT-compiled NTT matches eager execution."""
    N, q, _, psi_powers, twiddles = ntt_params
    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    rng = np.random.default_rng(DEFAULT_SEED)
    x = random_input(rng, q, shape=(batch, N))

    y_eager = student.ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)
    y_jit = jax.jit(
        lambda z: student.ntt(z, q=q, psi_powers=psi_powers, twiddles=twiddles)
    )(x)
    jax.block_until_ready(y_jit)

    np.testing.assert_array_equal(to_int64(y_eager), to_int64(y_jit))


def test_vmap_matches_direct(batch, ntt_params):
    """vmap over batch dimension matches direct batched call."""
    N, q, _, psi_powers, twiddles = ntt_params
    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    rng = np.random.default_rng(DEFAULT_SEED)
    x = random_input(rng, q, shape=(batch, N))

    y_direct = student.ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)
    y_vmapped = jax.vmap(
        lambda row: student.ntt(
            row[None, :], q=q, psi_powers=psi_powers, twiddles=twiddles
        )[0]
    )(x)
    jax.block_until_ready(y_vmapped)

    assert y_vmapped.shape == y_direct.shape
    np.testing.assert_array_equal(to_int64(y_vmapped), to_int64(y_direct))


# -----------------------------------------------------------------------------
# Algebraic Property Tests
# -----------------------------------------------------------------------------

def test_linearity(ntt_params):
    """NTT is linear: NTT(a + b) ≡ NTT(a) + NTT(b) (mod q)."""
    N, q, _, psi_powers, twiddles = ntt_params
    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    rng = np.random.default_rng(DEFAULT_SEED)
    a = random_input(rng, q, shape=(1, N))
    b = random_input(rng, q, shape=(1, N))

    left = student.ntt(
        (a + b) % q, q=q, psi_powers=psi_powers, twiddles=twiddles
    )
    right = (
        student.ntt(a, q=q, psi_powers=psi_powers, twiddles=twiddles)
        + student.ntt(b, q=q, psi_powers=psi_powers, twiddles=twiddles)
    ) % q

    np.testing.assert_array_equal(to_int64(left), to_int64(right))


def test_output_range_and_dtype(batch, ntt_params):
    """NTT outputs are uint32 in [0, q)."""
    N, q, _, psi_powers, twiddles = ntt_params
    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    rng = np.random.default_rng(DEFAULT_SEED)
    x = random_input(rng, q, shape=(batch, N))
    y = student.ntt(x, q=q, psi_powers=psi_powers, twiddles=twiddles)

    assert y.dtype == jnp.uint32
    y64 = to_int64(y)
    assert y64.min() >= 0
    assert y64.max() < q
