"""
Assignment 2 student implementation — v1: JIT Primitives.

Optimization over baseline:
  - @jax.jit applied to mod_add_32, mod_sub_32, mod_mul_32.
  - Eliminates Python dispatch overhead on every primitive call.
  - XLA can fuse consecutive elementwise ops within each JIT-compiled kernel.
  - Everything else is identical to baseline (no precomputation, same protocol order).
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 32-bit primitives — JIT compiled (v1 change)
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


def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """
    32-bit MLE update — still 3 kernel launches, but each is JIT-compiled.

    Formula: (one_eval - zero_eval) * target_eval + zero_eval  mod q
    """
    diff   = mod_sub_32(one_eval, zero_eval, q)
    scaled = mod_mul_32(diff, target_eval, q)
    return mod_add_32(scaled, zero_eval, q)


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# Helpers for sumcheck_32
# -----------------------------------------------------------------------------

def _eval_composition_32(var_at_t, expression, *, q):
    """
    Evaluate the polynomial expression pointwise over all (even, odd) pairs.

    var_at_t  : dict  var_name -> 1-D uint32 array of length `half`
    expression: list[list[str]]  outer = additive terms, inner = factors
    Returns   : 1-D uint32 array of length `half`
    """
    some_arr = next(iter(var_at_t.values()))
    total = jnp.zeros(some_arr.shape, dtype=jnp.uint32)

    for term in expression:
        prod = jnp.ones(some_arr.shape, dtype=jnp.uint32)
        for var_name in term:
            prod = mod_mul_32(prod, var_at_t[var_name], q)
        total = mod_add_32(total, prod, q)

    return total


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v1 (JIT primitives)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v1 32-bit SumCheck prover — JIT-compiled primitives.

    Only change from baseline: mod_add_32, mod_sub_32, mod_mul_32 are @jax.jit.
    Protocol order is identical — MLE table updates happen AFTER each round's
    evaluations are computed. No precomputation.
    """
    tables = {name: jnp.asarray(val, dtype=jnp.uint32)
              for name, val in eval_tables.items()}

    degree = max(len(term) for term in expression)
    q64 = jnp.asarray(q, dtype=jnp.uint64)

    all_round_evals = []

    for round_idx in range(num_rounds):
        evens = {name: tables[name][0::2] for name in tables}
        odds  = {name: tables[name][1::2] for name in tables}

        g_t_vals = []
        for t in range(degree + 1):
            if t == 0:
                var_at_t = evens
            elif t == 1:
                var_at_t = odds
            else:
                t_jax = jnp.asarray(t, dtype=jnp.uint32)
                var_at_t = {
                    name: mle_update_32(evens[name], odds[name], t_jax, q=q)
                    for name in tables
                }

            per_pair = _eval_composition_32(var_at_t, expression, q=q)
            g_t = (jnp.sum(per_pair.astype(jnp.uint64)) % q64).astype(jnp.uint32)
            g_t_vals.append(g_t)

        all_round_evals.append(g_t_vals)

        if round_idx < num_rounds - 1:
            r = challenges[round_idx]
            tables = {
                name: mle_update_32(evens[name], odds[name], r, q=q)
                for name in tables
            }

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
