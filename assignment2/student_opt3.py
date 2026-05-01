"""
Assignment 2 student implementation — v3: Stacked Tables.

Optimization over v2 (fused MLE):
  - eval_tables dict of V arrays replaced by a single (V, N) stacked array.
  - Variable string keys replaced by integer row indices.
  - eval_expression_stacked is @jax.jit with term_indices as static arg —
    JAX sees one contiguous tensor and can plan memory layout across all V
    variables simultaneously.
  - mle_update_32 applied to entire (V, N/2) stacked array in one call
    instead of per-variable calls.
  - Everything else identical to v2 — no precomputation, same protocol order.
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
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    raise NotImplementedError


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_64(a, b, q):
    """Optional 64-bit modular multiply kernel."""
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    """Optional 128-bit modular add kernel."""
    raise NotImplementedError


def mod_sub_128(a, b, q):
    """Optional 128-bit modular subtract kernel."""
    raise NotImplementedError


def mod_mul_128(a, b, q):
    """Optional 128-bit modular multiply kernel."""
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
    """
    Fused 32-bit MLE update — works on both 1-D (N,) and 2-D (V, N) arrays.

    For stacked tables, zero_eval/one_eval are (V, N/2) and the update is
    applied to all V variables in one kernel call.
    """
    q64    = jnp.uint64(q)
    z64    = zero_eval.astype(jnp.uint64)
    o64    = one_eval.astype(jnp.uint64)
    t64    = jnp.asarray(target_eval, dtype=jnp.uint64)
    diff   = (o64 + q64 - z64) % q64
    scaled = (diff * t64) % q64
    result = (scaled + z64) % q64
    return result.astype(jnp.uint32)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
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
# Stacked table helpers (v3)
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


@functools.partial(jax.jit, static_argnames=["term_indices"])
def _eval_expression_stacked(stacked_VN, q, term_indices):
    """
    Evaluate composed expression pointwise on stacked (V, N/2) tables.

    term_indices is static so JAX unrolls the loops at compile time.
    All V variables are accessed as integer row indices into one contiguous
    array — better cache reuse than V separate dict lookups.

    Returns: (N/2,) uint32 array
    """
    q64   = jnp.uint64(q)
    total = jnp.zeros(stacked_VN.shape[1], dtype=jnp.uint64)
    for term in term_indices:
        tv = stacked_VN[term[0]].astype(jnp.uint64)
        for idx in term[1:]:
            tv = (tv * stacked_VN[idx].astype(jnp.uint64)) % q64
        total = (total + tv) % q64
    return total.astype(jnp.uint32)


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v3 (stacked tables)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v3 32-bit SumCheck prover — stacked (V, N) tables with integer indices.

    Changes from v2:
      - dict replaced by (V, N) stacked array; string keys → integer row indices.
      - mle_update_32 applied to entire (V, N/2) block in one call.
      - eval_expression_stacked is @jax.jit with static term_indices.
    Protocol order unchanged — MLE updates after each round. No precomputation.
    """
    var_order    = sorted(eval_tables.keys())
    term_indices = _build_term_indices(expression, var_order)
    stacked      = _stack_tables(eval_tables, var_order)   # (V, N) uint32
    degree       = max(len(term) for term in expression)
    q64          = jnp.uint64(q)

    all_round_evals = []

    for round_idx in range(num_rounds):
        evens = stacked[:, 0::2]   # (V, N/2)
        odds  = stacked[:, 1::2]   # (V, N/2)

        g_t_vals = []
        for t in range(degree + 1):
            if t == 0:
                tables_t = evens
            elif t == 1:
                tables_t = odds
            else:
                # One fused MLE call updates all V rows simultaneously
                tables_t = mle_update_32(evens, odds, jnp.uint32(t), q)

            per_pair = _eval_expression_stacked(tables_t, q, term_indices)
            g_t = (jnp.sum(per_pair.astype(jnp.uint64)) % q64).astype(jnp.uint32)
            g_t_vals.append(g_t)

        all_round_evals.append(g_t_vals)

        if round_idx < num_rounds - 1:
            r       = challenges[round_idx]
            stacked = mle_update_32(evens, odds, r, q)   # (V, N/2)

    claim0 = mod_add_32(all_round_evals[0][0], all_round_evals[0][1], q)
    round_evals = jnp.stack([jnp.stack(g_t) for g_t in all_round_evals])

    return claim0, round_evals


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")
