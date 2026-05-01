"""
Benchmark runner for NTT latency measurement.

Usage:
    uv run python -m tests.benchmark
    uv run python -m tests.benchmark --tests
    uv run python -m tests.benchmark --bench
    uv run python -m tests.benchmark --logn 12 --batch 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

import provided
import student

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_LOGN = 10
DEFAULT_BATCH = 4
DEFAULT_RUNS = 20
DEFAULT_WARMUP = 5
DEFAULT_BIT_LENGTH = 31
DEFAULT_SEED = 42


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchStats:
    """Timing statistics for a single benchmark configuration."""
    compile_s: float
    median_s: float
    p90_s: float
    min_s: float
    max_s: float


# -----------------------------------------------------------------------------
# Benchmark Helpers
# -----------------------------------------------------------------------------

def summarize(compile_s, times):
    """
    Compute summary statistics from timing samples.

    Args:
        compile_s: JIT compilation time in seconds
        times: List of execution times in seconds

    Returns:
        BenchStats: Summary statistics
    """
    arr = np.asarray(times, dtype=np.float64)
    return BenchStats(
        compile_s=compile_s,
        median_s=float(np.median(arr)),
        p90_s=float(np.quantile(arr, 0.90)),
        min_s=float(np.min(arr)),
        max_s=float(np.max(arr)),
    )


def bench_single(N, batch, q, psi, runs, warmup, rng):
    """
    Benchmark NTT at a single (N, batch) configuration.

    Args:
        N: Transform size
        batch: Batch size
        q: Prime modulus
        psi: Primitive 2N-th root of unity
        runs: Number of timed iterations
        warmup: Number of warmup iterations (includes compile)
        rng: NumPy random generator

    Returns:
        BenchStats: Timing statistics
    """
    x_np = rng.integers(0, q, size=(batch, N), dtype=np.int64)
    x = jnp.asarray(x_np, dtype=jnp.uint32)

    psi_powers, twiddles = provided.precompute_tables(N, q, psi)
    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles = jnp.asarray(twiddles, dtype=jnp.uint32)

    prepare = getattr(student, "prepare_tables", None)
    if prepare is not None:
        psi_powers, twiddles = prepare(
            q=q, psi_powers=psi_powers, twiddles=twiddles
        )

    fn = jax.jit(
        lambda z: student.ntt(
            z, q=q, psi_powers=psi_powers, twiddles=twiddles
        )
    )

    # First call triggers JIT compilation
    t0 = time.perf_counter()
    y = fn(x)
    jax.block_until_ready(y)
    compile_s = time.perf_counter() - t0

    # Additional warmup
    for _ in range(max(0, warmup - 1)):
        y = fn(x)
        jax.block_until_ready(y)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        times.append(time.perf_counter() - t0)

    return summarize(compile_s, times)


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------

def run_correctness(logn, batch):
    """Run pytest test suite."""
    cmd = [
        "uv",
        "run",
        "pytest",
        "--logn",
        str(logn),
        "--batch",
        str(batch),
    ]
    subprocess.run(cmd, check=True)


def run_latency(logn, batch):
    """
    Run latency benchmarks and print results table.

    Args:
        logn: log2(N) for the benchmark
        batch: Batch size for the benchmark
    """
    console = Console()

    if logn <= 0:
        raise ValueError(f"logn must be positive, got {logn}")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")

    N = 1 << logn
    q = provided.generate_ntt_modulus(N, bit_length=DEFAULT_BIT_LENGTH)
    psi = provided.negacyclic_psi(N, q)
    rng = np.random.default_rng(DEFAULT_SEED)

    device = jax.devices()[0]
    console.print(f"Device: {device.platform} ({device.device_kind})")

    table = Table(title="Negacyclic NTT Latency")
    table.add_column("log₂(N)", justify="right")
    table.add_column("N", justify="right")
    table.add_column("compile (ms)", justify="right")
    table.add_column("median (ms)", justify="right")
    table.add_column("p90 (ms)", justify="right")
    table.add_column("Mcoeff/s", justify="right")

    stats = bench_single(
        N=N,
        batch=batch,
        q=q,
        psi=psi,
        runs=DEFAULT_RUNS,
        warmup=DEFAULT_WARMUP,
        rng=rng,
    )

    throughput = (batch * N) / stats.median_s / 1e6

    table.add_row(
        str(logn),
        str(N),
        f"{stats.compile_s * 1e3:.2f}",
        f"{stats.median_s * 1e3:.3f}",
        f"{stats.p90_s * 1e3:.3f}",
        f"{throughput:.2f}",
    )

    console.print(table)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser():
    """Build argument parser for benchmark CLI."""
    p = argparse.ArgumentParser(
        description="Run correctness tests and/or NTT latency benchmarks."
    )

    p.add_argument(
        "--tests",
        action="store_true",
        help="Run pytest suite",
    )
    p.add_argument(
        "--bench",
        action="store_true",
        help="Run latency benchmark",
    )
    p.add_argument(
        "--logn",
        type=int,
        default=DEFAULT_LOGN,
        help=f"log2(N) (default: {DEFAULT_LOGN})",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH})",
    )

    return p


def main():
    """Entry point for benchmark CLI."""
    args = build_parser().parse_args()

    run_tests = args.tests
    run_bench = args.bench
    if not run_tests and not run_bench:
        run_tests = True
        run_bench = True

    if run_tests:
        run_correctness(args.logn, args.batch)
    if run_bench:
        run_latency(args.logn, args.batch)


if __name__ == "__main__":
    main()
