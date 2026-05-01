"""
Assignment 2 student implementation — baseline (CPU, no optimizations).

Strategy:
  - 32-bit primitives upcast to uint64 to avoid overflow before mod.
  - mle_update_32: linear interpolation formula (o - z)*t + z mod q.
  - sumcheck_32: vectorized over pairs (no Python loop over table entries),
    Python loop only over rounds (num_rounds) and eval points (degree+1).
  - MLE table updates happen strictly AFTER computing each round's evals,
    using the provided challenge for that round. No precomputation.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return ((a64 + b64) % q64).astype(jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    # Add q before subtracting to keep the result non-negative in uint64.
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return ((a64 + q64 - b64) % q64).astype(jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    # a, b < q < 2^32  =>  a*b < 2^64, fits safely in uint64.
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return ((a64 * b64) % q64).astype(jnp.uint32)


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
    Compulsory 32-bit MLE update.

    Formula: (one_eval - zero_eval) * target_eval + zero_eval  mod q

    Works element-wise when zero_eval / one_eval are arrays and
    target_eval is a scalar, or all three are arrays of the same shape.
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
# SumCheck prover — 32-bit baseline
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    Baseline 32-bit SumCheck prover (CPU, no optimizations).

    Parameters
    ----------
    eval_tables : dict[str, jnp.ndarray]
        MLE evaluation tables (uint32), one per variable name.
    q           : jnp.uint32 scalar — prime modulus.
    expression  : list[list[str]] — polynomial to run SumCheck over.
    challenges  : 1-D jnp.uint32 array of length num_rounds-1.
                  Excludes the final verifier-only challenge.
    num_rounds  : int — total number of SumCheck rounds (== num_vars).

    Returns
    -------
    (claim0, round_evals)
      claim0      : jnp.uint32 scalar  = g_0(0) + g_0(1) mod q
      round_evals : jnp.uint32 array, shape [num_rounds, degree+1]
                    round_evals[i] = [g_i(0), g_i(1), ..., g_i(degree)]

    Protocol constraint satisfied
    ------------------------------
    MLE table updates (which halve the table size) are applied AFTER
    computing each round's evaluations, using challenges[round_idx].
    No precomputation of future rounds' tables.
    """
    # Working copies — all uint32.
    tables = {name: jnp.asarray(val, dtype=jnp.uint32)
              for name, val in eval_tables.items()}

    # Degree = length of longest multiplicative term in the expression.
    # e.g. a*b+c -> max(2,1) = 2, so we evaluate g(0), g(1), g(2).
    degree = max(len(term) for term in expression)

    q64 = jnp.asarray(q, dtype=jnp.uint64)  # for overflow-safe summation

    all_round_evals = []  # will become shape [num_rounds, degree+1]

    for round_idx in range(num_rounds):
        # ------------------------------------------------------------------ #
        # Step 1: Split current tables into even (x_i=0) and odd (x_i=1).   #
        # ------------------------------------------------------------------ #
        evens = {name: tables[name][0::2] for name in tables}
        odds  = {name: tables[name][1::2] for name in tables}

        # ------------------------------------------------------------------ #
        # Step 2: Compute g_i(t) for t = 0, 1, ..., degree.                 #
        #   t=0 -> use even entries directly                                 #
        #   t=1 -> use odd  entries directly                                 #
        #   t>=2 -> linearly extend via MLE update formula                   #
        # ------------------------------------------------------------------ #
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

            # Evaluate the polynomial expression pointwise over all pairs,
            # then sum. Each per-pair value < q < 2^32; half <= 2^19 pairs,
            # so the sum < 2^51, safe in uint64 before taking mod.
            per_pair = _eval_composition_32(var_at_t, expression, q=q)
            g_t = (jnp.sum(per_pair.astype(jnp.uint64)) % q64).astype(jnp.uint32)
            g_t_vals.append(g_t)

        all_round_evals.append(g_t_vals)

        # ------------------------------------------------------------------ #
        # Step 3: Apply MLE update to shrink tables for the NEXT round.      #
        #   Uses challenges[round_idx] (the challenge the verifier sent       #
        #   after seeing this round's evaluations).                           #
        #   Skipped for the last round — no challenge exists for it.         #
        # ------------------------------------------------------------------ #
        if round_idx < num_rounds - 1:
            r = challenges[round_idx]  # jnp.uint32 scalar from challenges array
            tables = {
                name: mle_update_32(evens[name], odds[name], r, q=q)
                for name in tables
            }

    # claim0 = initial sum = g_0(0) + g_0(1) mod q.
    claim0 = mod_add_32(all_round_evals[0][0], all_round_evals[0][1], q)

    # Pack into a 2-D JAX array [num_rounds, degree+1].
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
