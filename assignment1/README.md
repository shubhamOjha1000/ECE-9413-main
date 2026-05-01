# Assignment 1 — Negacyclic NTT in JAX

In this assignment, you will implement a **negacyclic** Number Theoretic 
Transform (NTT) in JAX and measure its performance. The goal is to keep 
the code correct and push performance as far as you can.

You are free (**and encouraged**) to use any AI tools, both for learning
this algorithm and optimizing your code. AI will certainly help, but you 
will likely need to understand and improve the output to reach top performance.

---

## Useful links

- [NTT Tutorial](https://eprint.iacr.org/2024/585.pdf)
- [Montgomery Multiplication Tutorial](https://cp-algorithms.com/algebra/montgomery_multiplication.html)
- [Fast GPU NTT #1](https://arxiv.org/pdf/2012.01968)
- [Fast GPU NTT #2](https://arxiv.org/pdf/2407.13055)
- NYU HPC GPU Access - TODO

---

## Algorithm specifics

The NTT is the FFT, but for exact modular arithmetic instead of complex numbers.
That makes it a core tool for fast polynomial multiplication, especially in
cryptography. 

The forward **negacyclic** NTT for polynomials modulo `x^N + 1`:

```
y[k] = sum_{n=0}^{N-1} x[n] * psi^{(2k+1)n}   (mod q)
```

Where:
* `N` is the transform size
* `q` is a prime modulus where `(q - 1)` is divisible by `2N`
* `psi` is a primitive `2N`-th root of unity modulo `q`, so `psi^N ≡ -1 (mod q)`

Your function must handle inputs shaped `(B, N)` (batch dimension `B`).
`psi_powers` and `twiddles` are inputs to your implementation. The tests pass
the tables from `provided.precompute_tables`, but you can transform them once
in `prepare_tables` or use your own layout internally. No auto-conversion is
required or expected.

You may use any correct NTT algorithm.

Optional hook:
`prepare_tables(q=..., psi_powers=..., twiddles=...)` can precompute or convert
tables once. The benchmark calls this before timing, so its cost is excluded.

---

## What to do

Your implementation goes in **`student.py`** — that's the only file you edit.

- Implement `ntt` in `student.py`.
- Implement `mod_add`, `mod_sub`, and `mod_mul` in `student.py`.
- Keep the public API unchanged so tests and benchmarks still run.
- Focus on speed. Correctness is required, but the goal is fast code.

Your implementation will **not** need to be modified to switch between CPU
and GPU backends. We suggest testing correctness _locally_. Once you have 
a working version, then migrate over to a GPU to test and optimize performance.

---

## Performance tips

- Keep everything in JAX and JIT the hot path.
- Precompute or convert tables once in `prepare_tables`.
- Modular arithmetic using `%` may not be the fastest approach.
- Use `uint32`/`uint64` carefully to avoid overflow and extra conversions.
- You can even use Pallas or other JAX lowering tools for additional performance.

---

## Setup

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c `
  "irm https://astral.sh/uv/install.ps1 | iex"
```

Then from this directory:

```bash
bash scripts/setup.sh
```

`scripts/setup.sh` installs CPU JAX by default. If an NVIDIA driver is present,
it installs the matching CUDA wheels. It does not modify your shell.

For GPU requirements, see:
https://docs.jax.dev/en/latest/installation.html

---

## Running tests

```bash
uv run pytest
uv run pytest --logn 10 --batch 4
```

---

## Running benchmarks

```bash
uv run python -m tests.benchmark
uv run python -m tests.benchmark --tests --logn 10 --batch 4
uv run python -m tests.benchmark --bench --logn 12 --batch 4
```

With no flags, it runs both tests and the benchmark.

Options:
```bash
uv run python -m tests.benchmark --tests            # only run tests
uv run python -m tests.benchmark --bench            # only run benchmark
```

---

## Submission

```bash
bash scripts/make_submission.sh
```

This runs tests and produces `code.zip`. Upload to Brightspace.
