"""
Assignment 2 student implementation — v6: JIT Full Round.

Optimization over v5 (vmap degree loop):
  - The entire round body (split + vmap g(t) + fold) wrapped in one @jax.jit.
  - XLA sees the full round as one computation graph and fuses everything.
  - Key win: diff = (odds - evens) mod q is now shared between:
      1. vmap: computing g(0)..g(d)
      2. fold: MLE update for next round
    Without this, diff was read from HBM twice per round (once in vmap, once in fold).
    With JIT full round, XLA keeps diff in registers/shared memory and reuses it.
  - Dispatch overhead: 5 Python dispatches/round → 1 dispatch/round.
  - Everything else identical to v5 — no precomputation, same protocol order.
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
# JIT full round factory (v6 key change)
# -----------------------------------------------------------------------------

def _make_jit_round(term_indices, degree):
    """
    Returns a single @jax.jit function covering one full SumCheck round:
      1. Reshape split (coalesced)
      2. diff computed ONCE
      3. vmap: g(t) for all t sharing diff
      4. Fold (MLE update) reusing same diff — no second HBM read

    XLA fuses all ops into as few GPU kernels as possible.
    """
    n_eval = degree + 1
    t_vals = jnp.arange(n_eval, dtype=jnp.uint32)

    @jax.jit
    def one_round(stacked, r, q):
        V, N_k = stacked.shape
        half   = N_k // 2
        q64    = jnp.uint64(q)

        # Coalesced split
        view  = stacked.reshape(V, half, 2)
        evens = view[:, :, 0].astype(jnp.uint64)   # (V, half)
        odds  = view[:, :, 1].astype(jnp.uint64)   # (V, half)

        # diff computed ONCE — reused by vmap AND fold below
        diff = (odds + q64 - evens) % q64           # (V, half)

        def g_at_t(t):
            tables_t = (diff * t.astype(jnp.uint64) % q64 + evens) % q64
            vals = jnp.zeros(half, dtype=jnp.uint64)
            for term in term_indices:
                tv = tables_t[term[0]]
                for idx in term[1:]:
                    tv = (tv * tables_t[idx]) % q64
                vals = (vals + tv) % q64
            return (jnp.sum(vals) % q64).astype(jnp.uint32)

        round_evals = jax.vmap(g_at_t)(t_vals)   # (n_eval,)

        # Fold: reuse diff — no re-read of evens/odds from HBM
        r64         = r.astype(jnp.uint64)
        new_stacked = (diff * r64 % q64 + evens) % q64   # (V, half)
        new_stacked = new_stacked.astype(jnp.uint32)

        return round_evals, new_stacked

    return one_round


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v6 (JIT full round)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v6 32-bit SumCheck prover — full round as a single JIT kernel.

    Changes from v5:
      - split + vmap + fold fused into one @jax.jit function.
      - diff shared between g(t) evals and the fold — saves one HBM read/round.
      - Python dispatch: 5 calls/round → 1 call/round.
    Protocol order unchanged — no precomputation.
    """
    var_order    = sorted(eval_tables.keys())
    term_indices = _build_term_indices(expression, var_order)
    stacked      = _stack_tables(eval_tables, var_order)   # (V, N) uint32
    degree       = max(len(term) for term in expression)

    one_round = _make_jit_round(term_indices, degree)
    all_round_evals = []

    for round_idx in range(num_rounds):
        # Use a dummy r=0 on the last round (fold result is discarded)
        r = challenges[round_idx] if round_idx < num_rounds - 1 else jnp.uint32(0)
        round_row, stacked = one_round(stacked, r, q)
        all_round_evals.append(round_row)

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
