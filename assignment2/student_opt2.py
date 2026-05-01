"""
Assignment 2 student implementation — v2: Fused MLE Update.

Optimization over v1 (JIT primitives):
  - mle_update_32 is now a single @jax.jit expression: (o-z)*t+z mod q.
  - XLA fuses sub+mul+add into one kernel — diff and scaled stay in registers,
    never written to HBM.
  - Reduces HBM traffic for MLE update from 9× to 3× (N/2 × 4 bytes):
      Unfused: read z, read o, write diff, read diff, write scaled,
               read scaled, read z, write result  → 9 passes
      Fused:   read z, read o, write result       → 3 passes
  - mod_add_32, mod_sub_32, mod_mul_32 remain @jax.jit from v1.
  - Everything else identical to v1 — no precomputation, same protocol order.
"""

from __future__ import annotations

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
    Fused 32-bit MLE update — sub + mul + add as ONE JIT expression.

    XLA keeps diff and scaled in registers, never written to HBM.
    1 kernel launch instead of 3, 3x less HBM traffic.

    Note: q is positional (not keyword) so jax.jit can trace it as a
    static scalar without needing static_argnames.
    """
    q64    = jnp.uint64(q)
    z64    = zero_eval.astype(jnp.uint64)
    o64    = one_eval.astype(jnp.uint64)
    t64    = jnp.asarray(target_eval, dtype=jnp.uint64)
    diff   = (o64 + q64 - z64) % q64       # (o - z) mod q
    scaled = (diff * t64) % q64             # diff * t mod q
    result = (scaled + z64) % q64           # + z mod q
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
# SumCheck prover — 32-bit v2 (fused MLE update)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v2 32-bit SumCheck prover — fused MLE update (1 kernel, 3x less HBM).

    Change from v1: mle_update_32 is a single @jax.jit expression.
    Protocol order unchanged — MLE table updates happen AFTER each round's
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
                    name: mle_update_32(evens[name], odds[name], t_jax, q)
                    for name in tables
                }

            per_pair = _eval_composition_32(var_at_t, expression, q=q)
            g_t = (jnp.sum(per_pair.astype(jnp.uint64)) % q64).astype(jnp.uint32)
            g_t_vals.append(g_t)

        all_round_evals.append(g_t_vals)

        if round_idx < num_rounds - 1:
            r = challenges[round_idx]
            tables = {
                name: mle_update_32(evens[name], odds[name], r, q)
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
