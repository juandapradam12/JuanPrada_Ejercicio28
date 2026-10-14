"""Supervised companion models for dengue prevalence (post-clustering).

Labels (`NoYes`) are used only here — never during unsupervised clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dengue import DEFAULT_FEATURES, DengueResult


@dataclass
class SupervisedResult:
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    holdout: dict[str, float]
    y_test: np.ndarray
    proba_test: dict[str, np.ndarray]
    pred_test: dict[str, np.ndarray]


def _pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def supervised_dengue_companion(
    result: DengueResult,
    features: list[str] | None = None,
    random_state: int = 42,
    n_splits: int = 5,
) -> SupervisedResult:
    """Cross-validated LR / RF on climate features → `NoYes`."""
    feats = list(features or DEFAULT_FEATURES)
    X = result.data[feats].to_numpy(dtype=float)
    y = result.data["NoYes"].astype(int).to_numpy()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, pipe in _pipelines(random_state).items():
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "accuracy": "accuracy",
                "f1": "f1",
            },
            n_jobs=-1,
        )
        rows.append(
            {
                "model": name,
                "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "roc_auc_std": float(np.std(scores["test_roc_auc"])),
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "accuracy_std": float(np.std(scores["test_accuracy"])),
                "f1_mean": float(np.mean(scores["test_f1"])),
                "f1_std": float(np.std(scores["test_f1"])),
            }
        )
    metrics = pd.DataFrame(rows)

    # Hold-out fit for curves / importances (same features, no cluster labels).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )
    proba_test: dict[str, np.ndarray] = {}
    pred_test: dict[str, np.ndarray] = {}
    fitted = {}
    for name, pipe in _pipelines(random_state).items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        proba_test[name] = pipe.predict_proba(X_test)[:, 1]
        pred_test[name] = pipe.predict(X_test)

    holdout = {
        "logistic_regression_roc_auc": float(roc_auc_score(y_test, proba_test["logistic_regression"])),
        "random_forest_roc_auc": float(roc_auc_score(y_test, proba_test["random_forest"])),
        "logistic_regression_accuracy": float(accuracy_score(y_test, pred_test["logistic_regression"])),
        "random_forest_accuracy": float(accuracy_score(y_test, pred_test["random_forest"])),
        "logistic_regression_f1": float(f1_score(y_test, pred_test["logistic_regression"])),
        "random_forest_f1": float(f1_score(y_test, pred_test["random_forest"])),
    }

    # Feature importance: |coef| for LR, impurity importance for RF
    lr = fitted["logistic_regression"].named_steps["model"]
    rf = fitted["random_forest"].named_steps["model"]
    importance = pd.DataFrame(
        {
            "feature": feats,
            "logistic_abs_coef": np.abs(lr.coef_.ravel()),
            "random_forest_importance": rf.feature_importances_,
        }
    ).sort_values("random_forest_importance", ascending=False)

    return SupervisedResult(
        metrics=metrics,
        feature_importance=importance,
        holdout=holdout,
        y_test=y_test,
        proba_test=proba_test,
        pred_test=pred_test,
    )


def plot_supervised_companion(
    supervised: SupervisedResult,
    out_dir: str | Path,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # CV metric bars
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(supervised.metrics))
    width = 0.35
    ax.bar(
        x - width / 2,
        supervised.metrics["roc_auc_mean"],
        width,
        yerr=supervised.metrics["roc_auc_std"],
        label="ROC-AUC",
        color="#2a6f97",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        supervised.metrics["f1_mean"],
        width,
        yerr=supervised.metrics["f1_std"],
        label="F1",
        color="#bc4749",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(supervised.metrics["model"].str.replace("_", "\n"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Supervised companion (5-fold CV on climate features → NoYes)")
    ax.legend()
    fig.tight_layout()
    paths["cv_metrics"] = out / "dengue_supervised_cv.png"
    fig.savefig(paths["cv_metrics"], dpi=150)
    plt.close(fig)

    # ROC curves on hold-out
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, color in (
        ("logistic_regression", "#2a6f97"),
        ("random_forest", "#bc4749"),
    ):
        RocCurveDisplay.from_predictions(
            supervised.y_test,
            supervised.proba_test[name],
            name=name.replace("_", " "),
            ax=ax,
            plot_chance_level=(name == "random_forest"),
        )
        # Style the most recently added line (API-stable across sklearn versions)
        if ax.lines:
            ax.lines[-1].set_color(color)
    ax.set_title("Hold-out ROC (labels unused in clustering)")
    fig.tight_layout()
    paths["roc"] = out / "dengue_supervised_roc.png"
    fig.savefig(paths["roc"], dpi=150)
    plt.close(fig)

    # Feature importance
    imp = supervised.feature_importance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(imp["feature"], imp["logistic_abs_coef"], color="#2a6f97")
    axes[0].invert_yaxis()
    axes[0].set_title("Logistic |coefficient|")
    axes[0].set_xlabel("|coef|")
    axes[1].barh(imp["feature"], imp["random_forest_importance"], color="#bc4749")
    axes[1].invert_yaxis()
    axes[1].set_title("Random forest importance")
    axes[1].set_xlabel("Importance")
    fig.suptitle("Which climate features drive dengue prediction?", y=1.02)
    fig.tight_layout()
    paths["importance"] = out / "dengue_feature_importance.png"
    fig.savefig(paths["importance"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix for best CV model by ROC-AUC
    best = supervised.metrics.loc[supervised.metrics["roc_auc_mean"].idxmax(), "model"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        supervised.y_test,
        supervised.pred_test[best],
        display_labels=["No", "Yes"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Hold-out confusion ({best.replace('_', ' ')})")
    fig.tight_layout()
    paths["confusion"] = out / "dengue_supervised_confusion.png"
    fig.savefig(paths["confusion"], dpi=150)
    plt.close(fig)

    return paths
