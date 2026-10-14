#!/usr/bin/env python3
"""Compare crude Monte Carlo vs variance-reduction estimators."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MPI_DIR = ROOT / "mpi"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

LINE_RE = re.compile(
    r"method=(?P<method>\w+)\s+N=(?P<n>\d+)\s+procs=(?P<procs>\d+)\s+"
    r"estimate=(?P<est>[0-9.eE+-]+)\s+exact=(?P<exact>[0-9.eE+-]+)\s+"
    r"rel_error=(?P<err>[0-9.eE+-]+)%\s+time=(?P<time>[0-9.eE+-]+)s"
)

METHODS = ("crude", "antithetic", "control", "importance")
NS = (1_000, 10_000, 100_000, 1_000_000)


def _launcher(mpiexec: str, procs: int) -> list[str]:
    launcher = shlex.split(mpiexec)
    if launcher and "mpi" in Path(launcher[0]).name and "--oversubscribe" not in launcher:
        launcher.append("--oversubscribe")
    return [*launcher, "-n", str(procs)]


def run_once(n: int, method: str, procs: int, mpiexec: str) -> dict:
    cmd = [*_launcher(mpiexec, procs), str(MPI_DIR / "integra"), str(n), method]
    env = os.environ.copy()
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    proc = subprocess.run(cmd, cwd=MPI_DIR, capture_output=True, text=True, env=env)
    match = LINE_RE.search(proc.stderr or "")
    if match is None:
        raise RuntimeError(
            f"Failed VR parse.\ncmd={cmd}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
        )
    return {
        "method": match.group("method"),
        "n_samples": int(match.group("n")),
        "procs": int(match.group("procs")),
        "estimate": float(match.group("est")),
        "rel_error_pct": float(match.group("err")),
        "time_s": float(match.group("time")),
    }


def compare(
    procs: int = 2,
    mpiexec: str = "mpirun",
    repeats: int = 3,
    ns: tuple[int, ...] = NS,
) -> pd.DataFrame:
    if not (MPI_DIR / "integra").exists():
        subprocess.run(["make", "integra"], cwd=MPI_DIR, check=True)

    rows = []
    for method in METHODS:
        for n in ns:
            for r in range(repeats):
                row = run_once(n, method, procs, mpiexec)
                row["repeat"] = r
                rows.append(row)
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["method", "n_samples"], as_index=False)
        .agg(
            rel_error_pct=("rel_error_pct", "mean"),
            rel_error_std=("rel_error_pct", "std"),
            time_s=("time_s", "mean"),
        )
        .sort_values(["method", "n_samples"])
    )
    return summary


def plot_comparison(summary: pd.DataFrame, out_dir: Path = FIG) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        sub = summary[summary["method"] == method]
        ax.errorbar(
            sub["n_samples"],
            sub["rel_error_pct"],
            yerr=sub["rel_error_std"].fillna(0),
            marker="o",
            capsize=3,
            label=method,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N samples")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(r"Variance reduction for $\int_0^1 e^{x}\sin(x)\,dx$")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "monte_carlo_variance_reduction.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpiexec = os.environ.get("MPIEXEC", "mpirun")
    summary = compare(procs=int(os.environ.get("NP", "2")), mpiexec=mpiexec, repeats=3)
    csv_path = OUT / "monte_carlo_vr_comparison.csv"
    summary.to_csv(csv_path, index=False)
    fig_path = plot_comparison(summary)
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")

    # Highlight best method at largest N
    largest = summary["n_samples"].max()
    at_n = summary[summary["n_samples"] == largest].sort_values("rel_error_pct")
    print("\nBest at N=%d: %s (%.4f%%)" % (largest, at_n.iloc[0]["method"], at_n.iloc[0]["rel_error_pct"]))


if __name__ == "__main__":
    main()
