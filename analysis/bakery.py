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
    hourly_item_rates: pd.DataFrame | None = None
    cluster_hour_share: pd.DataFrame | None = None
    frequent_itemsets: pd.DataFrame | None = None
    rule_engine: str = "apriori"


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

    # Attach temporal metadata (first timestamp seen per transaction)
    meta = (
        raw.groupby("Transaction", as_index=False)
        .agg(Date=("Date", "first"), Time=("Time", "first"))
    )
    baskets = baskets.merge(meta, on="Transaction", how="left")
    baskets["datetime"] = pd.to_datetime(
        baskets["Date"].astype(str) + " " + baskets["Time"].astype(str),
        errors="coerce",
    )
    baskets["hour"] = baskets["datetime"].dt.hour
    baskets["dayofweek"] = baskets["datetime"].dt.dayofweek
    baskets["day_name"] = baskets["datetime"].dt.day_name()
    baskets["month"] = baskets["datetime"].dt.month
    return baskets, item_cols


def temporal_item_rates(
    baskets: pd.DataFrame,
    item_cols: list[str],
    top_n: int = 8,
) -> pd.DataFrame:
    """Share of baskets containing each top item, by hour of day."""
    totals = baskets[item_cols].sum().sort_values(ascending=False)
    top_cols = list(totals.head(top_n).index)
    hourly = (
        baskets.dropna(subset=["hour"])
        .groupby("hour")[top_cols]
        .mean()
    )
    hourly.columns = [c.removeprefix("Item_") for c in hourly.columns]
    return hourly


def cluster_hour_distribution(baskets: pd.DataFrame) -> pd.DataFrame:
    """Row-normalized cluster mix by hour (purchase-occasion timing)."""
    tmp = baskets.dropna(subset=["hour", "cluster"]).copy()
    counts = pd.crosstab(tmp["hour"], tmp["cluster"])
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def _format_itemset(value) -> str:
    if isinstance(value, (set, frozenset)):
        return " & ".join(sorted(str(v) for v in value))
    return str(value)


def fp_growth_rules(
    baskets: pd.DataFrame,
    item_cols: list[str],
    min_support: float = 0.02,
    min_confidence: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FP-Growth frequent itemsets + association rules via mlxtend."""
    from mlxtend.frequent_patterns import association_rules as mx_rules
    from mlxtend.frequent_patterns import fpgrowth

    binary = baskets[item_cols].astype(bool).copy()
    binary.columns = [c.removeprefix("Item_") for c in binary.columns]
    frequent = fpgrowth(binary, min_support=min_support, use_colnames=True)
    if frequent.empty:
        empty_rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
        return empty_rules, frequent

    frequent = frequent.sort_values("support", ascending=False).reset_index(drop=True)
    rules = mx_rules(frequent, metric="confidence", min_threshold=min_confidence)
    if rules.empty:
        empty_rules = pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )
        return empty_rules, frequent

    out = pd.DataFrame(
        {
            "antecedents": rules["antecedents"].map(_format_itemset),
            "consequents": rules["consequents"].map(_format_itemset),
            "support": rules["support"].astype(float),
            "confidence": rules["confidence"].astype(float),
            "lift": rules["lift"].astype(float),
        }
    )
    out = out.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return out, frequent


def item_cooccurrence(
    baskets: pd.DataFrame,
    item_cols: list[str],
    top_n: int = 12,
) -> pd.DataFrame:
    """Pairwise co-occurrence counts among the most frequent items."""
    top = baskets[item_cols].sum().sort_values(ascending=False).head(top_n).index
    X = baskets[list(top)].to_numpy(dtype=np.float64)
    co = X.T @ X
    labels = [c.removeprefix("Item_") for c in top]
    return pd.DataFrame(co, index=labels, columns=labels)


def association_rules(
    baskets: pd.DataFrame,
    item_cols: list[str],
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    max_antecedent: int = 2,
) -> pd.DataFrame:
    """Lightweight Apriori-style pair/triple mining (fallback without mlxtend)."""
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

    rule_engine = "fp_growth"
    frequent_itemsets: pd.DataFrame | None
    try:
        rules, frequent_itemsets = fp_growth_rules(
            baskets,
            item_cols,
            min_support=min_support,
            min_confidence=min_confidence,
        )
    except ImportError:
        rule_engine = "apriori"
        frequent_itemsets = None
        rules = association_rules(
            baskets, item_cols, min_support=min_support, min_confidence=min_confidence
        )

    top_items = baskets[item_cols].sum().sort_values(ascending=False)
    top_items.index = top_items.index.str.removeprefix("Item_")
    hourly_item_rates = temporal_item_rates(baskets, item_cols)
    cluster_hour_share = cluster_hour_distribution(baskets)

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
        hourly_item_rates=hourly_item_rates,
        cluster_hour_share=cluster_hour_share,
        frequent_itemsets=frequent_itemsets,
        rule_engine=rule_engine,
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

    paths.update(plot_bakery_temporal(result, out))
    paths.update(plot_bakery_cooccurrence(result, out))
    return paths


def plot_bakery_cooccurrence(result: BakeryResult, out_dir: str | Path) -> dict[str, Path]:
    """Heatmap of top-item co-occurrence counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    co = item_cooccurrence(result.baskets, result.item_cols, top_n=12)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(co.to_numpy(), cmap="YlOrRd")
    ax.set_xticks(range(len(co.columns)))
    ax.set_yticks(range(len(co.index)))
    ax.set_xticklabels(co.columns, rotation=60, ha="right")
    ax.set_yticklabels(co.index)
    ax.set_title(f"Top-item co-occurrence ({result.rule_engine} rules companion)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="shared baskets")
    fig.tight_layout()
    path = out / "bakery_item_cooccurrence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"cooccurrence": path}


def plot_bakery_temporal(result: BakeryResult, out_dir: str | Path) -> dict[str, Path]:
    """Hour-of-day volume, item seasonality, and cluster timing plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    baskets = result.baskets.dropna(subset=["hour"])

    # Transactions by hour and by weekday
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    hour_counts = baskets["hour"].value_counts().sort_index()
    axes[0].bar(hour_counts.index.astype(int), hour_counts.values, color="#bc4749")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Transactions")
    axes[0].set_title("Traffic by hour")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = baskets["day_name"].value_counts().reindex(order).fillna(0)
    axes[1].bar(range(len(order)), day_counts.values, color="#2a6f97")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([d[:3] for d in order])
    axes[1].set_ylabel("Transactions")
    axes[1].set_title("Traffic by weekday")
    fig.tight_layout()
    paths["temporal_traffic"] = out / "bakery_temporal_traffic.png"
    fig.savefig(paths["temporal_traffic"], dpi=150)
    plt.close(fig)

    # Top-item attachment rate by hour (seasonality heatmap)
    if result.hourly_item_rates is not None and not result.hourly_item_rates.empty:
        rates = result.hourly_item_rates
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(rates.T.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(rates.columns)))
        ax.set_yticklabels(list(rates.columns))
        ax.set_xticks(range(len(rates.index)))
        ax.set_xticklabels([int(h) for h in rates.index])
        ax.set_xlabel("Hour of day")
        ax.set_title("Top-item basket rate by hour")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(item in basket)")
        fig.tight_layout()
        paths["hourly_items"] = out / "bakery_hourly_item_rates.png"
        fig.savefig(paths["hourly_items"], dpi=150)
        plt.close(fig)

    # Occasion mix by hour
    if result.cluster_hour_share is not None and not result.cluster_hour_share.empty:
        share = result.cluster_hour_share
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bottom = np.zeros(len(share))
        hours = share.index.to_numpy()
        for col in share.columns:
            vals = share[col].to_numpy()
            ax.bar(hours, vals, bottom=bottom, label=f"cluster {col}", width=0.9)
            bottom = bottom + vals
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Share of baskets")
        ax.set_ylim(0, 1)
        ax.set_title("Purchase-occasion mix by hour")
        ax.legend(ncols=min(len(share.columns), 4), fontsize=8, loc="upper right")
        fig.tight_layout()
        paths["cluster_hours"] = out / "bakery_cluster_by_hour.png"
        fig.savefig(paths["cluster_hours"], dpi=150)
        plt.close(fig)

    return paths
