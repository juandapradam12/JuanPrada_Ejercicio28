"""Bakery basket PCA, clustering, and association-rule analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import Normalizer


@dataclass
class BakeryResult:
    baskets: pd.DataFrame
    item_cols: list[str]
    metrics: pd.DataFrame
    best_k: int
    labels: np.ndarray
    pca: PCA
    embedding: np.ndarray
    rules: pd.DataFrame
    top_items: pd.Series
    svd_components: int
    primary_split_k: int | None = None


def load_baskets(
    path: str | Path,
    drop_items: Iterable[str] = ("NONE", "Adjustment"),
) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(path)
    raw["Item"] = raw["Item"].astype(str).str.strip()
    raw = raw[~raw["Item"].isin(set(drop_items))].copy()

    # One row per transaction with binary item indicators
    baskets = (
        raw.groupby(["Transaction", "Item"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
        .astype(np.uint8)
    )
    baskets.columns = [f"Item_{c}" for c in baskets.columns]
    item_cols = list(baskets.columns)
    baskets = baskets.reset_index()
    return baskets, item_cols


def association_rules(
    baskets: pd.DataFrame,
    item_cols: list[str],
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    max_antecedent: int = 2,
) -> pd.DataFrame:
    """Lightweight Apriori-style pair/triple mining (no extra dependency)."""
    X = baskets[item_cols].to_numpy(dtype=np.uint8)
    n = X.shape[0]
    support_1 = X.mean(axis=0)
    frequent = {item_cols[i]: float(support_1[i]) for i in np.where(support_1 >= min_support)[0]}
    if not frequent:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    freq_idx = [item_cols.index(c) for c in frequent]
    rows = []

    # 2-item rules
    for i, j in combinations(freq_idx, 2):
        both = np.logical_and(X[:, i], X[:, j]).mean()
        if both < min_support:
            continue
        for a, c in ((i, j), (j, i)):
            conf = both / support_1[a]
            if conf < min_confidence:
                continue
            lift = conf / support_1[c]
            rows.append(
                {
                    "antecedents": item_cols[a].removeprefix("Item_"),
                    "consequents": item_cols[c].removeprefix("Item_"),
                    "support": float(both),
                    "confidence": float(conf),
                    "lift": float(lift),
                }
            )

    # Optional 3-item itemsets -> single-consequent rules
    if max_antecedent >= 2 and len(freq_idx) <= 40:
        for i, j, k in combinations(freq_idx, 3):
            trip = np.logical_and.reduce((X[:, i], X[:, j], X[:, k])).mean()
            if trip < min_support:
                continue
            for ant, cons in (
                ((i, j), k),
                ((i, k), j),
                ((j, k), i),
            ):
                ant_sup = np.logical_and(X[:, ant[0]], X[:, ant[1]]).mean()
                if ant_sup <= 0:
                    continue
                conf = trip / ant_sup
                if conf < min_confidence:
                    continue
                lift = conf / support_1[cons]
                rows.append(
                    {
                        "antecedents": (
                            f"{item_cols[ant[0]].removeprefix('Item_')}"
                            f" & {item_cols[ant[1]].removeprefix('Item_')}"
                        ),
                        "consequents": item_cols[cons].removeprefix("Item_"),
                        "support": float(trip),
                        "confidence": float(conf),
                        "lift": float(lift),
                    }
                )

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    return rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def _inertia_knee(ks: np.ndarray, inertia: np.ndarray) -> int:
    """K at maximum distance from the inertia curve to the endpoints chord."""
    x = ks.astype(float)
    y = np.log(np.maximum(inertia, 1e-12))
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = np.hypot(x2 - x1, y2 - y1) + 1e-12
    # 2D point-line distance (NumPy 2.x no longer accepts 2-vectors in np.cross).
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    return int(ks[int(np.argmax(distances))])


def _select_occasion_k(metrics: pd.DataFrame) -> tuple[int, int | None]:
    """
    Choose an interpretable purchase-occasion K.

    Silhouette on basket embeddings often rises slowly with K; prefer the
    inertia knee among K>=3, breaking ties with Calinski–Harabasz.
    """
    sil_best = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    usable = metrics[metrics["k"] >= 3].copy()
    if usable.empty:
        return sil_best, None

    knee = _inertia_knee(usable["k"].to_numpy(), usable["inertia"].to_numpy())
    # Re-rank a small window around the knee by Calinski–Harabasz / silhouette.
    window = usable[(usable["k"] >= knee - 2) & (usable["k"] <= knee + 2)]
    if "calinski_harabasz" in window.columns and window["calinski_harabasz"].notna().any():
        best_k = int(window.loc[window["calinski_harabasz"].idxmax(), "k"])
    else:
        best_k = int(window.loc[window["silhouette"].idxmax(), "k"])
    primary = sil_best if sil_best != best_k else None
    return best_k, primary


def analyze_bakery(
    path: str | Path,
    k_min: int = 2,
    k_max: int = 15,
    random_state: int = 42,
    min_support: float = 0.02,
    min_confidence: float = 0.35,
    svd_components: int = 25,
) -> BakeryResult:
    baskets, item_cols = load_baskets(path)
    X_raw = baskets[item_cols].to_numpy(dtype=float)

    # Sparse binary baskets → normalize + TruncatedSVD before K-Means.
    # This avoids a trivial K=2 split dominated by a single popular item in
    # ambient dimension (~90), and yields more stable occasion geometry.
    n_comp = min(svd_components, X_raw.shape[1] - 1)
    X_norm = Normalizer(norm="l2").fit_transform(X_raw)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X = svd.fit_transform(X_norm)

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
            }
        )
    metrics = pd.DataFrame(rows)
    best_k, primary_split_k = _select_occasion_k(metrics)

    model = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
    labels = model.fit_predict(X)
    baskets = baskets.copy()
    baskets["cluster"] = labels

    pca = PCA(n_components=2, random_state=random_state)
    embedding = pca.fit_transform(X)
    rules = association_rules(
        baskets, item_cols, min_support=min_support, min_confidence=min_confidence
    )
    top_items = baskets[item_cols].sum().sort_values(ascending=False)
    top_items.index = top_items.index.str.removeprefix("Item_")

    return BakeryResult(
        baskets=baskets,
        item_cols=item_cols,
        metrics=metrics,
        best_k=best_k,
        labels=labels,
        pca=pca,
        embedding=embedding,
        rules=rules,
        top_items=top_items,
        svd_components=n_comp,
        primary_split_k=primary_split_k,
    )


def cluster_item_profile(result: BakeryResult, cluster_id: int, top_n: int = 12) -> pd.Series:
    mask = result.baskets["cluster"] == cluster_id
    profile = result.baskets.loc[mask, result.item_cols].sum().sort_values(ascending=False).head(top_n)
    profile.index = profile.index.str.removeprefix("Item_")
    return profile


def plot_bakery_diagnostics(result: BakeryResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(result.metrics["k"], result.metrics["inertia"], "o-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow (basket K-Means)")
    axes[1].plot(result.metrics["k"], result.metrics["silhouette"], "o-")
    axes[1].axvline(result.best_k, color="crimson", ls="--", label=f"selected K={result.best_k}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Silhouette vs K")
    axes[1].legend()
    fig.tight_layout()
    paths["diagnostics"] = out / "bakery_cluster_diagnostics.png"
    fig.savefig(paths["diagnostics"], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        result.embedding[:, 0],
        result.embedding[:, 1],
        c=result.labels,
        cmap="tab20",
        s=10,
        alpha=0.7,
    )
    ax.set_xlabel(f"PC1 ({result.pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({result.pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Bakery baskets in PCA space")
    fig.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    paths["pca"] = out / "bakery_pca_clusters.png"
    fig.savefig(paths["pca"], dpi=150)
    plt.close(fig)

    top = result.top_items.head(15)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(top.index.astype(str), top.values, color="#bc4749")
    ax.set_ylabel("Transactions containing item")
    ax.set_title("Most frequent bakery items")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    paths["top_items"] = out / "bakery_top_items.png"
    fig.savefig(paths["top_items"], dpi=150)
    plt.close(fig)

    profile = cluster_item_profile(result, cluster_id=int(pd.Series(result.labels).mode().iloc[0]))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(profile.index.astype(str), profile.values, color="#6a994e")
    ax.set_title("Example purchase occasion (largest cluster)")
    ax.set_ylabel("Item counts")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    paths["occasion"] = out / "bakery_example_occasion.png"
    fig.savefig(paths["occasion"], dpi=150)
    plt.close(fig)

    return paths
