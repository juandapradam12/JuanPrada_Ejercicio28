#!/usr/bin/env python3
"""Run dengue + bakery unsupervised analyses and write figures/tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.bakery import analyze_bakery, plot_bakery_diagnostics
from analysis.dengue import analyze_dengue, external_validation, plot_dengue_diagnostics
from analysis.supervised import plot_supervised_companion, supervised_dengue_companion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--skip-supervised",
        action="store_true",
        help="Skip the post-clustering supervised dengue companion models",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    dengue = analyze_dengue(args.data_dir / "dengue.csv")
    dengue_paths = plot_dengue_diagnostics(dengue, args.figures_dir)
    dengue_ext = external_validation(dengue)
    dengue.metrics.to_csv(args.out_dir / "dengue_cluster_metrics.csv", index=False)
    dengue.data.to_csv(args.out_dir / "dengue_with_clusters.csv", index=False)

    supervised_summary = None
    if not args.skip_supervised:
        supervised = supervised_dengue_companion(dengue)
        supervised_paths = plot_supervised_companion(supervised, args.figures_dir)
        dengue_paths.update(supervised_paths)
        supervised.metrics.to_csv(args.out_dir / "dengue_supervised_metrics.csv", index=False)
        supervised.feature_importance.to_csv(
            args.out_dir / "dengue_feature_importance.csv", index=False
        )
        supervised_summary = {
            "cv_metrics": supervised.metrics.to_dict(orient="records"),
            "holdout": supervised.holdout,
            "top_feature": supervised.feature_importance.iloc[0].to_dict(),
        }

    bakery = analyze_bakery(args.data_dir / "BreadBasket_DMS.csv")
    bakery_paths = plot_bakery_diagnostics(bakery, args.figures_dir)
    bakery.metrics.to_csv(args.out_dir / "bakery_cluster_metrics.csv", index=False)
    bakery.rules.head(50).to_csv(args.out_dir / "bakery_association_rules.csv", index=False)
    bakery.baskets[["Transaction", "cluster", "hour", "dayofweek", "day_name"]].to_csv(
        args.out_dir / "bakery_transaction_clusters.csv", index=False
    )
    if bakery.hourly_item_rates is not None:
        bakery.hourly_item_rates.to_csv(args.out_dir / "bakery_hourly_item_rates.csv")
    if bakery.cluster_hour_share is not None:
        bakery.cluster_hour_share.to_csv(args.out_dir / "bakery_cluster_hour_share.csv")

    peak_hour = None
    if "hour" in bakery.baskets.columns and bakery.baskets["hour"].notna().any():
        peak_hour = int(bakery.baskets["hour"].value_counts().idxmax())

    summary = {
        "dengue": {
            "n_regions": int(len(dengue.data)),
            "best_k": int(dengue.best_k),
            "pca_variance_ratio": dengue.pca.explained_variance_ratio_.tolist(),
            "external_validation": dengue_ext,
            "supervised_companion": supervised_summary,
            "figures": {k: str(v) for k, v in dengue_paths.items()},
        },
        "bakery": {
            "n_transactions": int(len(bakery.baskets)),
            "n_items": int(len(bakery.item_cols)),
            "best_k": int(bakery.best_k),
            "primary_split_k": bakery.primary_split_k,
            "svd_components": int(bakery.svd_components),
            "n_association_rules": int(len(bakery.rules)),
            "top_rule": bakery.rules.iloc[0].to_dict() if len(bakery.rules) else None,
            "peak_hour": peak_hour,
            "pca_variance_ratio": bakery.pca.explained_variance_ratio_.tolist(),
            "figures": {k: str(v) for k, v in bakery_paths.items()},
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
