# ECE 9413 Assignments — Shubham Ojha

Implementations and report for two GPU-accelerated cryptographic primitives in
JAX:

- **Assignment 1**: Negacyclic Number Theoretic Transform (NTT) — `assignment1/`
- **Assignment 2**: SumCheck protocol prover — `assignment2/`
- **Final report**: `report/Final_Report (1).pdf` (LaTeX source: `report/main.tex`)

The submitted student code is `assignment1/student.py` and
`assignment2/student.py`. Per-assignment READMEs document the assignment-level
commands; this top-level README is the shortest reproducible path to verify the
report's headline numbers.

---

## Hardware used in the report

- **Official benchmark runs**: NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09,
  CUDA 13.0, JAX/JAXLIB 0.10.0.
- **Optimization exploration**: NVIDIA A100-SXM4-40GB.
- The H100 was a rented GPU; `ncu` was blocked by `ERR_NVGPUCTRPERM`, so all
  mechanism claims are grounded in HLO operation counts and end-to-end timing,
  not hardware performance counters.

---

## Quickstart: verify required runs

Both assignments use `uv` for dependency management. Install once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Assignment 1 — NTT

```bash
cd assignment1
bash scripts/setup.sh

# Required correctness
uv run pytest
uv run pytest --logn 10 --batch 4
uv run python -m tests.benchmark --tests --logn 10 --batch 4

# Headline H100 official benchmark (10,499 Mcoeff/s in the report)
uv run python -m tests.benchmark --bench --logn 20 --batch 4
```
