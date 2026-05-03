"""
Assignment 2 student implementation — v7: JIT Full SumCheck.

Optimization over v6 (JIT full round):
  - The entire sumcheck_32 function including the outer Python for-loop is
    wrapped in @jax.jit via a factory function.
  - JAX unrolls the Python for-loop at trace time → one giant XLA program
    covering all num_rounds rounds.
  - Zero Python overhead at runtime: all n rounds execute inside one XLA call
    (1 dispatch instead of n dispatches).
  - XLA can optimize across rounds — pipeline memory accesses between rounds.

  Trade-off:
  - Compile time is O(n) — grows linearly with num_rounds.
  - For very large n (>=24), use lax.scan instead (constant compile time).
  - For vars4/16/20 (n=4/16/20) compile time is acceptable.
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


def mle_update_32(zero_eval, one_eval, target_eval, q):
    """Fused 32-bit MLE update."""
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
# Full JIT sumcheck factory (v7 key change)
# -----------------------------------------------------------------------------

# Cache compiled functions per (expression, num_rounds) to avoid retracing.
_sumcheck_cache = {}

def _make_sumcheck_jit(expression, num_rounds):
    """
    Factory: bakes expression and num_rounds as static values at trace time.
    Returns (jit_fn, var_order).

    The Python for-loop over num_rounds is unrolled at trace time by JAX →
    one XLA program covering all rounds. Zero Python overhead at runtime.
    """
    key = (tuple(tuple(t) for t in expression), num_rounds)
    if key in _sumcheck_cache:
        return _sumcheck_cache[key]

    var_order    = sorted(set(v for term in expression for v in term))
    term_indices = _build_term_indices(expression, var_order)
    degree       = max(len(term) for term in expression)
    n_eval       = degree + 1
    t_vals       = jnp.arange(n_eval, dtype=jnp.uint32)

    @jax.jit
    def _sumcheck(stacked, q, challenges):
        """
        stacked:    (V, 2^num_rounds) uint32 — tables in var_order row order
        q:          uint32 scalar
        challenges: (num_rounds-1,) uint32
        """
        q64 = jnp.uint64(q)
        V   = stacked.shape[0]
        all_round_evals = []

        # Python for-loop — unrolled at trace time into one XLA program
        for round_idx in range(num_rounds):
            N_k  = stacked.shape[1]
            half = N_k // 2

            # Coalesced split
            view  = stacked.reshape(V, half, 2)
            evens = view[:, :, 0].astype(jnp.uint64)
            odds  = view[:, :, 1].astype(jnp.uint64)

            # diff shared by vmap AND fold
            diff = (odds + q64 - evens) % q64

            def g_at_t(t):
                tables_t = (diff * t.astype(jnp.uint64) % q64 + evens) % q64
                vals = jnp.zeros(half, dtype=jnp.uint64)
                for term in term_indices:
                    tv = tables_t[term[0]]
                    for idx in term[1:]:
                        tv = (tv * tables_t[idx]) % q64
                    vals = (vals + tv) % q64
                return (jnp.sum(vals) % q64).astype(jnp.uint32)

            round_row = jax.vmap(g_at_t)(t_vals)   # (n_eval,)
            all_round_evals.append(round_row)

            # Fold — reuses diff, no second HBM read
            if round_idx < num_rounds - 1:
                r64     = challenges[round_idx].astype(jnp.uint64)
                stacked = ((diff * r64 % q64 + evens) % q64).astype(jnp.uint32)

        round_evals = jnp.stack(all_round_evals)   # (num_rounds, n_eval)
        claim0 = ((round_evals[0, 0].astype(jnp.uint64) +
                   round_evals[0, 1].astype(jnp.uint64)) % q64).astype(jnp.uint32)
        return claim0, round_evals

    _sumcheck_cache[key] = (_sumcheck, var_order)
    return _sumcheck, var_order


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v7 (JIT full sumcheck)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v7 32-bit SumCheck prover — entire function JIT'd, loop unrolled at trace time.

    Changes from v6:
      - Outer Python for-loop over num_rounds included inside @jax.jit.
      - JAX unrolls all rounds at trace time → one XLA program, 1 dispatch.
      - Compile time is O(num_rounds) — acceptable for n=4/16/20.
    Protocol order unchanged — no precomputation.
    """
    sc_fn, var_order = _make_sumcheck_jit(expression, num_rounds)
    stacked = _stack_tables(eval_tables, var_order)   # (V, N) uint32
    return sc_fn(stacked, q, challenges)


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
