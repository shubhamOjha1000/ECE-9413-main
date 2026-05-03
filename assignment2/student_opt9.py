"""
Assignment 2 student implementation — v9: Production (auto-select v7 vs v8).

Final implementation combining all optimizations:
  1. Stacked (V, N) SoA layout (v3)
  2. Reshape split for coalesced GPU access (v4)
  3. diff computed once, shared by vmap AND fold (v6)
  4. vmap over degree evaluation points (v5)
  5. Full JIT — zero Python overhead at runtime (v7/v8)
  6. Auto-selects backend:
       num_rounds <= 20 → JIT unroll (v7): faster runtime, XLA optimizes across rounds
       num_rounds >  20 → lax.scan  (v8): O(1) compile time, constant regardless of n
  7. Compile cache: each (expression, num_rounds) pair compiled only once.
"""

from __future__ import annotations

import jax
import jax.lax as lax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 32-bit primitives — JIT compiled
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
# v7 backend: JIT unroll (fast runtime, O(n) compile)
# -----------------------------------------------------------------------------

def _make_jit_unroll(expression, num_rounds, term_indices, var_order):
    degree = max(len(term) for term in expression)
    n_eval = degree + 1

    @jax.jit
    def _fn(stacked, q, challenges):
        t_vals = jnp.arange(n_eval, dtype=jnp.uint32)
        q64 = jnp.uint64(q)
        V   = stacked.shape[0]
        all_round_evals = []

        for round_idx in range(num_rounds):
            N_k  = stacked.shape[1]
            half = N_k // 2
            view  = stacked.reshape(V, half, 2)
            evens = view[:, :, 0].astype(jnp.uint64)
            odds  = view[:, :, 1].astype(jnp.uint64)
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

            round_row = jax.vmap(g_at_t)(t_vals)
            all_round_evals.append(round_row)

            if round_idx < num_rounds - 1:
                r64     = challenges[round_idx].astype(jnp.uint64)
                stacked = ((diff * r64 % q64 + evens) % q64).astype(jnp.uint32)

        round_evals = jnp.stack(all_round_evals)
        claim0 = ((round_evals[0, 0].astype(jnp.uint64) +
                   round_evals[0, 1].astype(jnp.uint64)) % q64).astype(jnp.uint32)
        return claim0, round_evals

    return _fn


# -----------------------------------------------------------------------------
# v8 backend: lax.scan (O(1) compile, fixed-buffer carry)
# -----------------------------------------------------------------------------

def _make_lax_scan(expression, num_rounds, term_indices, var_order):
    degree = max(len(term) for term in expression)
    n_eval = degree + 1

    @jax.jit
    def _fn(stacked, q, challenges):
        t_vals = jnp.arange(n_eval, dtype=jnp.uint32)
        q64  = jnp.uint64(q)
        V, N = stacked.shape

        r_padded = jnp.concatenate([
            challenges,
            jnp.zeros(num_rounds - challenges.shape[0], dtype=jnp.uint32)
        ])

        def round_body(carry, r):
            buffer, active_len = carry
            half = active_len // 2

            active      = lax.dynamic_slice_in_dim(buffer, 0, active_len, axis=1)
            active_view = active.reshape(V, half, 2)
            evens = active_view[:, :, 0].astype(jnp.uint64)
            odds  = active_view[:, :, 1].astype(jnp.uint64)
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

            round_evals_row = jax.vmap(g_at_t)(t_vals)

            r64        = r.astype(jnp.uint64)
            new_active = ((diff * r64 % q64 + evens) % q64).astype(jnp.uint32)
            new_buffer = lax.dynamic_update_slice_in_dim(buffer, new_active, 0, axis=1)

            return (new_buffer, half), round_evals_row

        _, all_round_evals = lax.scan(round_body, (stacked, N), r_padded)

        claim0 = ((all_round_evals[0, 0].astype(jnp.uint64) +
                   all_round_evals[0, 1].astype(jnp.uint64)) % q64).astype(jnp.uint32)
        return claim0, all_round_evals

    return _fn


# -----------------------------------------------------------------------------
# Compile cache + auto-selector
# -----------------------------------------------------------------------------

_COMPILED_CACHE = {}

def _get_compiled(expression, num_rounds):
    """
    Return cached (fn, var_order) for this (expression, num_rounds).
    Auto-selects v7 (JIT unroll) for n<=20, v8 (lax.scan) for n>20.
    """
    cache_key = (tuple(tuple(t) for t in expression), num_rounds)
    if cache_key in _COMPILED_CACHE:
        return _COMPILED_CACHE[cache_key]

    var_order    = sorted(set(v for term in expression for v in term))
    term_indices = _build_term_indices(expression, var_order)

    if num_rounds <= 20:
        fn = _make_jit_unroll(expression, num_rounds, term_indices, var_order)
    else:
        fn = _make_lax_scan(expression, num_rounds, term_indices, var_order)

    _COMPILED_CACHE[cache_key] = (fn, var_order)
    return fn, var_order


# -----------------------------------------------------------------------------
# SumCheck prover — 32-bit v9 (production)
# -----------------------------------------------------------------------------

def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """
    v9 production 32-bit SumCheck prover — all optimizations combined.

    Auto-selects:
      num_rounds <= 20 → JIT unroll (v7): faster runtime
      num_rounds >  20 → lax.scan  (v8): O(1) compile time
    Compiled functions are cached per (expression, num_rounds).
    Protocol order unchanged — no precomputation.
    """
    fn, var_order = _get_compiled(expression, num_rounds)
    stacked = _stack_tables(eval_tables, var_order)
    return fn(stacked, q, challenges)


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
