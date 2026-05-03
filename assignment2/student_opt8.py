"""
Assignment 2 student implementation — v8: lax.scan Outer Loop.

Optimization over v7 (JIT full sumcheck):
  - Python for-loop over num_rounds replaced by jax.lax.scan.
  - v7 unrolled all n rounds at trace time → compile time O(n).
  - v8 compiles only ONE round body and repeats it n times at runtime
    → compile time O(1) regardless of num_rounds.

  The shape problem and solution:
  - lax.scan requires carry to have FIXED shape at every step.
  - Tables shrink each round: (V,N) → (V,N/2) → (V,N/4) — violates fixed shape.
  - Solution: fixed-size buffer (V, N) + active_len scalar as carry.
    Each round reads only the first active_len columns, folds to active_len//2,
    writes result back into the first half of the same buffer.

  Trade-off vs v7:
  - v7 wins at small n (XLA optimizes across rounds, faster runtime)
  - v8 wins at large n (constant compile time, better for n > 20)
"""

from __future__ import annotations

import jax
import jax.lax as lax
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
# lax.scan sumcheck factory (v8 key change)
# -----------------------------------------------------------------------------

_scan_cache = {}

def _make_sumcheck_scan(expression, num_rounds):
    """
    Factory: returns a JIT + lax.scan compiled sumcheck.
    Compile time is O(1) in num_rounds — only the round body is compiled once.
    """
    key = (tuple(tuple(t) for t in expression), num_rounds)
    if key in _scan_cache:
        return _scan_cache[key]

    var_order    = sorted(set(v for term in expression for v in term))
    term_indices = _build_term_indices(expression, var_order)
    degree       = max(len(term) for term in expression)
    n_eval       = degree + 1

    @jax.jit
    def sumcheck_scan(stacked, q, challenges):
        t_vals = jnp.arange(n_eval, dtype=jnp.uint32)
        """
        stacked:    (V, N) uint32 where N = 2^num_rounds
        q:          uint32 scalar
        challenges: (num_rounds-1,) uint32 — padded to (num_rounds,) internally
        """
        q64  = jnp.uint64(q)
        V, N = stacked.shape

        # Pad challenges to length num_rounds (last element unused — fold skipped)
        r_padded = jnp.concatenate([
            challenges,
            jnp.zeros(num_rounds - challenges.shape[0], dtype=jnp.uint32)
        ])

        half = N // 2  # static — N is concrete from stacked.shape

        def round_body(buf, r):
            # buf: (V, N) — valid data packed at front, rest zeros
            # Reshape always uses static half = N//2; zero pairs contribute 0
            view  = buf.reshape(V, half, 2)
            evens = view[:, :, 0].astype(jnp.uint64)   # (V, half)
            odds  = view[:, :, 1].astype(jnp.uint64)   # (V, half)
            diff  = (odds + q64 - evens) % q64

            def g_at_t(t):
                tables_t = (diff * t.astype(jnp.uint64) % q64 + evens) % q64
                vals = jnp.zeros(half, dtype=jnp.uint64)
                for term in term_indices:
                    tv = tables_t[term[0]]
                    for idx in term[1:]:
                        tv = (tv * tables_t[idx]) % q64
                    vals = (vals + tv) % q64
                return (jnp.sum(vals) % q64).astype(jnp.uint32)

            round_evals_row = jax.vmap(g_at_t)(t_vals)   # (n_eval,)

            r64        = r.astype(jnp.uint64)
            new_active = ((diff * r64 % q64 + evens) % q64).astype(jnp.uint32)  # (V, half)
            # Pad fold result back to (V, N): zero pairs stay zero next round
            new_buf = jnp.concatenate(
                [new_active, jnp.zeros((V, half), dtype=jnp.uint32)], axis=1
            )

            return new_buf, round_evals_row

        _, all_round_evals = lax.scan(round_body, stacked, r_padded)
        # all_round_evals: (num_rounds, n_eval)

        claim0 = ((all_round_evals[0, 0].astype(jnp.uint64) +
                   all_round_evals[0, 1].astype(jnp.uint64)) % q64).astype(jnp.uint32)
        return claim0, all_round_evals

    _scan_cache[key] = (sumcheck_scan, var_order)
    return sumcheck_scan, var_order


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v8 (lax.scan)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v8 32-bit SumCheck prover — lax.scan outer loop, O(1) compile time.

    Changes from v7:
      - Python for-loop replaced by lax.scan.
      - Fixed-size buffer carry (V, N) + active_len solves the shrinking shape problem.
      - Compile time is constant regardless of num_rounds.
      - Runtime may be slightly slower than v7 for small n (XLA can't optimize across rounds).
    Protocol order unchanged — no precomputation.
    """
    sc_fn, var_order = _make_sumcheck_scan(expression, num_rounds)
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
