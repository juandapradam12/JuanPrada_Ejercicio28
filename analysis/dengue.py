"""Dengue prevalence clustering and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["humid", "temp", "h10pix", "trees", "Xmin", "Ymin"]


@dataclass
class DengueResult:
    data: pd.DataFrame
    features: list[str]
    X_scaled: np.ndarray
    metrics: pd.DataFrame
    best_k: int
    labels: np.ndarray
    pca: PCA
    embedding: np.ndarray


def load_dengue(path: str | Path, features: Iterable[str] = DEFAULT_FEATURES) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "rownames" in df.columns:
        df = df.drop(columns=["rownames"])
    feats = list(features)
    missing = [c for c in feats + ["NoYes"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dengue data: {missing}")
    return df.dropna(subset=feats + ["NoYes"]).reset_index(drop=True)


def cluster_model_scores(
    X: np.ndarray,
    k_values: range | list[int],
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        km_labels = km.fit_predict(X)
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gmm_labels = gmm.fit_predict(X)
        agg = AgglomerativeClustering(n_clusters=k)
        agg_labels = agg.fit_predict(X)

        for name, labels, inertia in (
            ("kmeans", km_labels, km.inertia_),
            ("gmm", gmm_labels, np.nan),
            ("agglomerative", agg_labels, np.nan),
        ):
            rows.append(
                {
                    "model": name,
                    "k": k,
                    "inertia": inertia,
                    "silhouette": silhouette_score(X, labels),
                    "calinski_harabasz": calinski_harabasz_score(X, labels),
                }
            )
    return pd.DataFrame(rows)


def pick_best_k(metrics: pd.DataFrame, model: str = "kmeans") -> int:
    subset = metrics[metrics["model"] == model]
    return int(subset.loc[subset["silhouette"].idxmax(), "k"])


def analyze_dengue(
    path: str | Path,
    features: Iterable[str] = DEFAULT_FEATURES,
    k_min: int = 2,
    k_max: int = 12,
    random_state: int = 42,
) -> DengueResult:
    data = load_dengue(path, features)
    feats = list(features)
    scaler = StandardScaler()
    X = scaler.fit_transform(data[feats].to_numpy())

    metrics = cluster_model_scores(X, range(k_min, k_max + 1), random_state=random_state)
    best_k = pick_best_k(metrics, model="kmeans")

    model = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
    labels = model.fit_predict(X)
    data = data.copy()
    data["cluster"] = labels

    pca = PCA(n_components=2, random_state=random_state)
    embedding = pca.fit_transform(X)

    return DengueResult(
        data=data,
        features=feats,
        X_scaled=X,
        metrics=metrics,
        best_k=best_k,
        labels=labels,
        pca=pca,
        embedding=embedding,
    )


def external_validation(result: DengueResult) -> dict[str, float]:
    y = result.data["NoYes"].astype(int).to_numpy()
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y, result.labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, result.labels)),
        "dengue_rate_by_cluster": float(
            result.data.groupby("cluster")["NoYes"].mean().std(ddof=0)
        ),
    }


def plot_dengue_diagnostics(result: DengueResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    km = result.metrics[result.metrics["model"] == "kmeans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(km["k"], km["inertia"], "o-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow (K-Means)")
    axes[1].plot(km["k"], km["silhouette"], "o-", label="K-Means")
    for model in ("gmm", "agglomerative"):
        sub = result.metrics[result.metrics["model"] == model]
        axes[1].plot(sub["k"], sub["silhouette"], "o--", label=model.upper() if model == "gmm" else "Agglomerative")
    axes[1].axvline(result.best_k, color="crimson", ls="--", alpha=0.8, label=f"selected K={result.best_k}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Model comparison")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    paths["diagnostics"] = out / "dengue_cluster_diagnostics.png"
    fig.savefig(paths["diagnostics"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        result.embedding[:, 0],
        result.embedding[:, 1],
        c=result.labels,
        cmap="tab10",
        s=18,
        alpha=0.8,
    )
    ax.set_xlabel(f"PC1 ({result.pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({result.pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Dengue regions in PCA space")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    paths["pca"] = out / "dengue_pca_clusters.png"
    fig.savefig(paths["pca"], dpi=150)
    plt.close(fig)

    profile = result.data.groupby("cluster")[result.features + ["NoYes"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    profile["NoYes"].plot(kind="bar", ax=ax, color="#2a6f97")
    ax.set_ylabel("Mean dengue prevalence (NoYes)")
    ax.set_title("Cluster risk profile")
    ax.set_xlabel("Cluster")
    fig.tight_layout()
    paths["risk"] = out / "dengue_cluster_risk.png"
    fig.savefig(paths["risk"], dpi=150)
    plt.close(fig)

    paths.update(plot_dengue_geo(result, out))
    return paths


def plot_dengue_geo(result: DengueResult, out_dir: str | Path) -> dict[str, Path]:
    """Map clusters and historical prevalence in longitude/latitude space."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    df = result.data

    # Prefer region midpoints when both bounds exist; fall back to minima.
    lon = (
        (df["Xmin"] + df["Xmax"]) / 2.0
        if {"Xmin", "Xmax"}.issubset(df.columns)
        else df["Xmin"]
    )
    lat = (
        (df["Ymin"] + df["Ymax"]) / 2.0
        if {"Ymin", "Ymax"}.issubset(df.columns)
        else df["Ymin"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc0 = axes[0].scatter(
        lon,
        lat,
        c=df["cluster"],
        cmap="tab10",
        s=14,
        alpha=0.85,
        linewidths=0,
    )
    axes[0].set_title("Unsupervised climate/land-cover clusters")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(sc0, ax=axes[0], label="cluster", fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(
        lon,
        lat,
        c=df["NoYes"],
        cmap="coolwarm",
        s=14,
        alpha=0.85,
        linewidths=0,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Historical dengue presence (NoYes)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(sc1, ax=axes[1], label="NoYes", fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", alpha=0.4)

    fig.suptitle("Dengue regions in geographic space", y=1.02)
    fig.tight_layout()
    paths["geo"] = out / "dengue_geo_clusters.png"
    fig.savefig(paths["geo"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths
