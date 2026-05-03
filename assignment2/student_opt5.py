"""
Assignment 2 student implementation — v5: vmap Degree Loop.

Optimization over v4 (reshape split):
  - The inner for-loop over t=0..degree replaced by jax.vmap.
  - All g(0)..g(d) computed in ONE batched kernel call instead of d+1 sequential calls.
  - diff = (odds - evens) mod q computed ONCE and shared across all t values:
      Without vmap: recompute diff for each t → V * (degree-1) HBM reads
      With vmap:    compute diff once, broadcast over t → V * 1 HBM read
  - Reshape split (v4) retained for coalesced GPU memory access.
  - Everything else identical to v4 — no precomputation, same protocol order.
"""

from __future__ import annotations

import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 32-bit primitives — JIT compiled (carried over from v1)
# -----------------------------------------------------------------------------

@jax.jit
def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return ((a.astype(jnp.uint64) + b.astype(jnp.uint64)) % q64).astype(jnp.uint32)


@jax.jit
def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return ((a.astype(jnp.uint64) + q64 - b.astype(jnp.uint64)) % q64).astype(jnp.uint32)


@jax.jit
def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return ((a.astype(jnp.uint64) * b.astype(jnp.uint64)) % q64).astype(jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    raise NotImplementedError

def mod_sub_64(a, b, q):
    raise NotImplementedError

def mod_mul_64(a, b, q):
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    raise NotImplementedError

def mod_sub_128(a, b, q):
    raise NotImplementedError

def mod_mul_128(a, b, q):
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


@jax.jit
def mle_update_32(zero_eval, one_eval, target_eval, q):
    """Fused 32-bit MLE update — works on (V, N) stacked arrays."""
    q64    = jnp.uint64(q)
    z64    = zero_eval.astype(jnp.uint64)
    o64    = one_eval.astype(jnp.uint64)
    t64    = jnp.asarray(target_eval, dtype=jnp.uint64)
    diff   = (o64 + q64 - z64) % q64
    scaled = (diff * t64) % q64
    result = (scaled + z64) % q64
    return result.astype(jnp.uint32)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError

def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    raise NotImplementedError

def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# Stacked table helpers
# -----------------------------------------------------------------------------

def _stack_tables(eval_tables, var_order):
    """Convert dict of 1-D arrays to (V, N) stacked uint32 array."""
    return jnp.stack(
        [jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in var_order],
        axis=0,
    )


def _build_term_indices(expression, var_order):
    """Convert expression list[list[str]] to tuple[tuple[int]] of row indices."""
    vi = {v: i for i, v in enumerate(var_order)}
    return tuple(tuple(vi[v] for v in term) for term in expression)


# -----------------------------------------------------------------------------
# vmap round evaluator factory (v5 key change)
# -----------------------------------------------------------------------------

def _make_round_evaluator(term_indices, degree):
    """
    Returns a JIT+vmap compiled function that computes g(0)..g(degree)
    in one batched call, sharing diff across all t values.
    """
    n_eval = degree + 1
    t_vals = jnp.arange(n_eval, dtype=jnp.uint32)   # [0, 1, ..., degree]

    def _g_at_t_from_diff(t, evens64, diff64, q64):
        """Compute g(t) given pre-computed diff = (odds - evens) mod q."""
        tables_t = (diff64 * t.astype(jnp.uint64) % q64 + evens64) % q64
        vals = jnp.zeros(tables_t.shape[1], dtype=jnp.uint64)
        for term in term_indices:
            tv = tables_t[term[0]]
            for idx in term[1:]:
                tv = (tv * tables_t[idx]) % q64
            vals = (vals + tv) % q64
        return (jnp.sum(vals) % q64).astype(jnp.uint32)

    @jax.jit
    def compute_all_g(evens, odds, q):
        """
        Compute g(0)..g(degree) simultaneously.
        evens, odds: (V, N//2) uint32
        Returns: (degree+1,) uint32
        """
        q64     = jnp.uint64(q)
        evens64 = evens.astype(jnp.uint64)
        odds64  = odds.astype(jnp.uint64)
        # diff computed ONCE, shared across all t values
        diff64  = (odds64 + q64 - evens64) % q64

        batched = jax.vmap(lambda t: _g_at_t_from_diff(t, evens64, diff64, q64))
        return batched(t_vals)   # (n_eval,) uint32

    return compute_all_g


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v5 (vmap degree loop)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v5 32-bit SumCheck prover — vmap over degree loop.

    Changes from v4:
      - for t in range(degree+1) replaced by jax.vmap over t_vals array.
      - diff = (odds - evens) computed once per round, shared across all t.
      - Reshape split (v4) retained.
    Protocol order unchanged — no precomputation.
    """
    var_order    = sorted(eval_tables.keys())
    term_indices = _build_term_indices(expression, var_order)
    stacked      = _stack_tables(eval_tables, var_order)   # (V, N)
    degree       = max(len(term) for term in expression)
    V            = stacked.shape[0]
    q64          = jnp.uint64(q)

    compute_all_g = _make_round_evaluator(term_indices, degree)
    all_round_evals = []

    for round_idx in range(num_rounds):
        N_k   = stacked.shape[1]
        view  = stacked.reshape(V, N_k // 2, 2)
        evens = view[:, :, 0]   # (V, N_k//2) coalesced
        odds  = view[:, :, 1]   # (V, N_k//2) coalesced

        # All g(t) in one batched call
        round_evals_row = compute_all_g(evens, odds, q)   # (degree+1,)
        all_round_evals.append(round_evals_row)

        if round_idx < num_rounds - 1:
            r       = challenges[round_idx]
            stacked = mle_update_32(evens, odds, r, q)   # (V, N_k//2)

    claim0 = mod_add_32(all_round_evals[0][0], all_round_evals[0][1], q)
    round_evals = jnp.stack(all_round_evals)   # (num_rounds, degree+1)

    return claim0, round_evals


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    raise NotImplementedError

def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    raise NotImplementedError

def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables, q=q, expression=expression,
            challenges=challenges, num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables, q=q, expression=expression,
            challenges=challenges, num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables, q=q, expression=expression,
            challenges=challenges, num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")
