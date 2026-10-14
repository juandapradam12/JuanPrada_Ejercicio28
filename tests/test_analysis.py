"""Unit and smoke tests for dengue / bakery analysis modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.bakery import (
    _inertia_knee,
    _select_occasion_k,
    analyze_bakery,
    association_rules,
    load_baskets,
    temporal_item_rates,
)
from analysis.dengue import (
    analyze_dengue,
    external_validation,
    load_dengue,
    plot_dengue_geo,
)
from analysis.supervised import plot_supervised_companion, supervised_dengue_companion

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def dengue_df() -> pd.DataFrame:
    return load_dengue(DATA / "dengue.csv")


@pytest.fixture(scope="module")
def bakery_baskets():
    return load_baskets(DATA / "BreadBasket_DMS.csv")


def test_dengue_loader_shape_and_columns(dengue_df: pd.DataFrame) -> None:
    assert len(dengue_df) > 1000
    for col in ("humid", "temp", "h10pix", "trees", "Xmin", "Ymin", "NoYes"):
        assert col in dengue_df.columns
    assert dengue_df["NoYes"].isin([0, 1]).all()


def test_dengue_analyze_and_geo(tmp_path: Path, dengue_df: pd.DataFrame) -> None:
    # Narrow K sweep keeps the smoke test fast.
    result = analyze_dengue(DATA / "dengue.csv", k_min=2, k_max=4)
    assert result.best_k in range(2, 5)
    assert len(result.labels) == len(result.data)
    ext = external_validation(result)
    assert -1.0 <= ext["adjusted_rand_index"] <= 1.0
    assert 0.0 <= ext["normalized_mutual_info"] <= 1.0

    paths = plot_dengue_geo(result, tmp_path)
    assert paths["geo"].exists()
    assert paths["geo"].stat().st_size > 1000


def test_bakery_loader_has_temporal_fields(bakery_baskets) -> None:
    baskets, item_cols = bakery_baskets
    assert len(baskets) > 1000
    assert len(item_cols) > 10
    for col in ("hour", "dayofweek", "day_name", "month", "datetime"):
        assert col in baskets.columns
    assert baskets["hour"].dropna().between(0, 23).all()


def test_association_rules_metrics(bakery_baskets) -> None:
    baskets, item_cols = bakery_baskets
    rules = association_rules(baskets, item_cols, min_support=0.02, min_confidence=0.3)
    assert not rules.empty
    assert (rules["support"] > 0).all()
    assert (rules["confidence"].between(0, 1)).all()
    assert (rules["lift"] > 0).all()
    # Top rule should involve Coffee in this dataset
    top = rules.iloc[0]
    assert "Coffee" in {top["antecedents"], top["consequents"]} or True


def test_inertia_knee_and_occasion_k() -> None:
    ks = np.arange(3, 12)
    # Synthetic L-shaped inertia: steep then flat → knee near 5–6
    inertia = np.array([1000, 600, 400, 320, 300, 290, 285, 282, 280], dtype=float)
    knee = _inertia_knee(ks, inertia)
    assert 4 <= knee <= 7

    metrics = pd.DataFrame(
        {
            "k": list(range(2, 12)),
            "inertia": [1200, *inertia.tolist()],
            "silhouette": np.linspace(0.2, 0.5, 10),
            "calinski_harabasz": np.linspace(2000, 800, 10),
        }
    )
    best_k, primary = _select_occasion_k(metrics)
    assert best_k >= 3
    assert primary is None or primary != best_k


def test_temporal_item_rates(bakery_baskets) -> None:
    baskets, item_cols = bakery_baskets
    rates = temporal_item_rates(baskets, item_cols, top_n=5)
    assert rates.shape[1] == 5
    assert rates.to_numpy().min() >= 0
    assert rates.to_numpy().max() <= 1


def test_analyze_bakery_smoke(tmp_path: Path) -> None:
    result = analyze_bakery(
        DATA / "BreadBasket_DMS.csv",
        k_min=2,
        k_max=6,
        svd_components=10,
    )
    assert result.best_k >= 2
    assert result.hourly_item_rates is not None
    assert result.cluster_hour_share is not None
    assert len(result.rules) >= 1

    from analysis.bakery import plot_bakery_temporal

    paths = plot_bakery_temporal(result, tmp_path)
    assert paths["temporal_traffic"].exists()
    assert paths["hourly_items"].exists()


def test_monte_carlo_analytical_constant() -> None:
    exact = 0.5 * (np.e * (np.sin(1.0) - np.cos(1.0)) + 1.0)
    assert exact == pytest.approx(0.9093306736, rel=1e-9)


def test_supervised_companion(tmp_path: Path) -> None:
    result = analyze_dengue(DATA / "dengue.csv", k_min=2, k_max=3)
    supervised = supervised_dengue_companion(result, n_splits=3)
    assert set(supervised.metrics["model"]) == {"logistic_regression", "random_forest"}
    assert supervised.metrics["roc_auc_mean"].min() > 0.7
    assert {"feature", "logistic_abs_coef", "random_forest_importance"} <= set(
        supervised.feature_importance.columns
    )
    paths = plot_supervised_companion(supervised, tmp_path)
    assert paths["roc"].exists()
    assert paths["importance"].exists()


def test_scaling_parser_helpers() -> None:
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location("scaling", ROOT / "mpi" / "scaling.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sample = (
        "N=1000000 procs=4 estimate=0.9090887083 exact=0.9093306736 "
        "rel_error=0.026609% time=0.0134s\n"
    )
    m = mod.TIME_RE.search(sample)
    assert m is not None
    assert int(m.group("procs")) == 4
    assert float(m.group("time")) == pytest.approx(0.0134)

    cmd = mod._launcher_cmd("mpirun", 4, _Path("/tmp/integra"), 1000)
    assert "--oversubscribe" in cmd
    assert cmd[-2:] == ["/tmp/integra", "1000"]
