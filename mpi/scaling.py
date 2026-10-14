#!/usr/bin/env python3
"""Strong-scaling study for the Monte Carlo integrator."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MPI_DIR = ROOT / "mpi"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

TIME_RE = re.compile(
    r"(?:method=\w+\s+)?"
    r"N=(?P<n>\d+)\s+procs=(?P<procs>\d+)\s+estimate=(?P<est>[0-9.eE+-]+)\s+"
    r"exact=(?P<exact>[0-9.eE+-]+)\s+rel_error=(?P<err>[0-9.eE+-]+)%\s+"
    r"time=(?P<time>[0-9.eE+-]+)s"
)


def _launcher_cmd(mpiexec: str, procs: int, binary: Path, n_samples: int) -> list[str]:
    """Build an mpirun/mpiexec command that works on small CI VMs."""
    launcher = shlex.split(mpiexec)
    # Open MPI on 2-core GitHub runners rejects -n 4 unless oversubscribed.
    if launcher and "mpi" in Path(launcher[0]).name and "--oversubscribe" not in launcher:
        launcher.append("--oversubscribe")
    return [*launcher, "-n", str(procs), str(binary), str(n_samples)]


def run_once(n_samples: int, procs: int, mpiexec: str) -> dict:
    cmd = _launcher_cmd(mpiexec, procs, MPI_DIR / "integra", n_samples)
    # Open MPI in CI/root environments often needs this flag.
    env = os.environ.copy()
    if "OMPI_ALLOW_RUN_AS_ROOT" not in env:
        env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=MPI_DIR,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    stderr = proc.stderr or ""
    match = TIME_RE.search(stderr)
    if match is None:
        raise RuntimeError(
            f"Failed to parse timing from: {cmd}\n"
            f"stdout={proc.stdout!r}\nstderr={stderr!r}\ncode={proc.returncode}"
        )
    return {
        "n_samples": int(match.group("n")),
        "procs": int(match.group("procs")),
        "estimate": float(match.group("est")),
        "rel_error_pct": float(match.group("err")),
        "time_s": float(match.group("time")),
    }


def run_scaling(
    n_samples: int = 5_000_000,
    procs_list: list[int] | None = None,
    mpiexec: str = "mpirun",
    repeats: int = 2,
) -> pd.DataFrame:
    procs_list = procs_list or [1, 2, 4]
    # Ensure binary exists
    binary = MPI_DIR / "integra"
    if not binary.exists():
        subprocess.run(["make", "integra"], cwd=MPI_DIR, check=True)

    rows = []
    for p in procs_list:
        for r in range(repeats):
            row = run_once(n_samples, p, mpiexec)
            row["repeat"] = r
            rows.append(row)
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("procs", as_index=False)
        .agg(
            n_samples=("n_samples", "first"),
            time_s=("time_s", "mean"),
            time_std=("time_s", "std"),
            rel_error_pct=("rel_error_pct", "mean"),
        )
        .sort_values("procs")
    )
    t1 = float(summary.loc[summary["procs"] == summary["procs"].min(), "time_s"].iloc[0])
    p_min = int(summary["procs"].min())
    summary["speedup"] = t1 / summary["time_s"]
    summary["ideal_speedup"] = summary["procs"] / p_min
    summary["efficiency"] = summary["speedup"] / summary["ideal_speedup"]
    return summary


def plot_scaling(summary: pd.DataFrame, out_dir: Path = FIG) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].errorbar(
        summary["procs"],
        summary["time_s"],
        yerr=summary["time_std"].fillna(0),
        fmt="o-",
        color="#2a6f97",
        capsize=4,
        label="measured",
    )
    axes[0].set_xlabel("MPI ranks")
    axes[0].set_ylabel("Wall time (s)")
    axes[0].set_title(f"Strong scaling (N={int(summary['n_samples'].iloc[0]):,})")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[0].legend()

    axes[1].plot(summary["procs"], summary["speedup"], "o-", color="#bc4749", label="speedup")
    axes[1].plot(summary["procs"], summary["ideal_speedup"], "k--", label="ideal")
    axes[1].set_xlabel("MPI ranks")
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Speedup vs ideal")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].legend()

    fig.tight_layout()
    path = out_dir / "mpi_strong_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5_000_000)
    parser.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mpiexec", default=os.environ.get("MPIEXEC", "mpirun"))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = run_scaling(
        n_samples=args.n_samples,
        procs_list=args.procs,
        mpiexec=args.mpiexec,
        repeats=args.repeats,
    )
    csv_path = OUT / "mpi_scaling.csv"
    summary.to_csv(csv_path, index=False)
    fig_path = plot_scaling(summary)
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
