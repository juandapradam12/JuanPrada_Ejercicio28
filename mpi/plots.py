#!/usr/bin/env python3
"""Plot relative Monte Carlo error vs number of samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

NS = np.array([100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000], dtype=np.int64)
EXACT = 0.5 * (np.e * (np.sin(1.0) - np.cos(1.0)) + 1.0)


def load_estimates(ns: np.ndarray) -> np.ndarray:
    estimates = []
    for n in ns:
        path = OUT / f"mc_{n}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `make -C mpi convergence` first."
            )
        estimates.append(float(path.read_text().strip().splitlines()[0]))
    return np.asarray(estimates, dtype=float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    estimates = load_estimates(NS)
    rel_err = np.abs(estimates - EXACT) / EXACT * 100.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(NS, rel_err, "o-", label="Monte Carlo relative error")
    ax.loglog(NS, 100.0 / np.sqrt(NS), "--", label=r"$O(N^{-1/2})$ reference")
    ax.set_xlabel(r"$N$ samples")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(r"Monte Carlo convergence for $\int_0^1 e^{x}\sin(x)\,dx$")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    out_path = FIG / "monte_carlo_relative_error.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"Exact integral ≈ {EXACT:.10f}")
    for n, est, err in zip(NS, estimates, rel_err):
        print(f"N={n:<10d} estimate={est:.8f} rel_err={err:.4f}%")


if __name__ == "__main__":
    main()
