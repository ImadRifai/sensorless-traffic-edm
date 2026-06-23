"""
Congestion model: a gradient-boosted classifier that predicts a four-level
congestion class (low / medium / high / peak) for a road segment from purely
*static* attributes (lanes, speed limit, length, road class, structure flags).

It is trained on the minority of Valencia segments that sit under an
electromagnetic loop sensor and then applied to the ~98% of streets that have
no sensor — and, by transfer, to a town with none at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .pipeline import (
    BINARY_FEATURES,
    CLASS_NAMES,
    HIGHWAY_CATEGORIES,
    NUMERIC_FEATURES,
    build_feature_frame,
    feature_columns,
)

# Hyper-parameters carried over from the notebook's tuned XGBoost.
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    learning_rate=0.03,
    n_estimators=600,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    reg_alpha=0.5,
    random_state=42,
    n_jobs=-1,
)


@dataclass
class CongestionModel:
    """Self-contained, serialisable bundle: estimator + everything to reuse it."""

    estimator: Pipeline
    highway_categories: list[str]
    qcut_bins: list[float]
    feature_names: list[str]
    priors: dict = field(default_factory=dict)

    # ---- inference ----------------------------------------------------- #
    def predict_level(self, clean_df: pd.DataFrame) -> np.ndarray:
        X = build_feature_frame(clean_df, self.highway_categories)
        idx = self.estimator.predict(X)
        return np.asarray(CLASS_NAMES)[idx.astype(int)]

    def predict_proba(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        X = build_feature_frame(clean_df, self.highway_categories)
        proba = self.estimator.predict_proba(X)
        return pd.DataFrame(proba, columns=CLASS_NAMES, index=clean_df.index)

    def bin_label(self, i: int) -> str:
        lo, hi = self.qcut_bins[i], self.qcut_bins[i + 1]
        lo = max(lo, 0)
        return f"{lo:.0f}–{hi:.0f} veh/h"

    # ---- persistence --------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> "CongestionModel":
        return joblib.load(Path(path))


def make_target(vehicles_per_hour: pd.Series) -> tuple[pd.Series, list[float]]:
    """Quartile-based ordinal target (low/medium/high/peak) + bin edges."""
    codes, bins = pd.qcut(vehicles_per_hour, q=4, labels=[0, 1, 2, 3], retbins=True)
    return codes.astype(int), [float(b) for b in bins]


def _build_estimator() -> Pipeline:
    # XGBoost is scale-invariant, but we keep the notebook's scaling of the
    # numeric block for fidelity; the rest passes through untouched.
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), NUMERIC_FEATURES)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("clf", XGBClassifier(**XGB_PARAMS))])


def train_and_evaluate(labeled: pd.DataFrame, target_col: str = "vehiculos_por_hora") -> dict:
    """
    Fit on a held-out split to report honest metrics, run 5-fold CV, then refit
    on all labeled data for deployment. Returns the model bundle + a metrics dict.
    """
    df = labeled.dropna(subset=[target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    y, bins = make_target(df[target_col])
    X = build_feature_frame(df, HIGHWAY_CATEGORIES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    est = _build_estimator()
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    proba_test = est.predict_proba(X_test)

    cv = cross_val_score(
        _build_estimator(), X, y, cv=5, scoring="f1_macro", n_jobs=-1
    )

    roc, macro_auc = _roc_ovr(np.asarray(y_test), proba_test)
    calibration = _calibration(np.asarray(y_test), proba_test)

    metrics = {
        "n_labeled": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "test_macro_auc": macro_auc,
        "cv_f1_macro_mean": float(cv.mean()),
        "cv_f1_macro_std": float(cv.std()),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "class_thresholds": [round(max(b, 0), 1) for b in bins],
        "roc": roc,
        "calibration": calibration,
    }

    # Deployed model: refit on every labeled segment.
    final_est = _build_estimator()
    final_est.fit(X, y)

    importances = _feature_importance(final_est)
    metrics["feature_importance"] = importances

    model = CongestionModel(
        estimator=final_est,
        highway_categories=HIGHWAY_CATEGORIES,
        qcut_bins=bins,
        feature_names=feature_columns(HIGHWAY_CATEGORIES),
    )
    return {"model": model, "metrics": metrics}


def _feature_importance(est: Pipeline) -> list[dict]:
    clf = est.named_steps["clf"]
    # After the ColumnTransformer the numeric block comes first (same order),
    # then the passthrough block in original order.
    passthrough = [c for c in feature_columns(HIGHWAY_CATEGORIES) if c not in NUMERIC_FEATURES]
    names = NUMERIC_FEATURES + passthrough
    imp = clf.feature_importances_
    pairs = sorted(zip(names, imp), key=lambda t: t[1], reverse=True)
    return [{"feature": n, "importance": float(v)} for n, v in pairs]


def _roc_ovr(y_true: np.ndarray, proba: np.ndarray) -> tuple[list[dict], float]:
    """One-vs-rest ROC curves + per-class AUC, plus the macro-average AUC.

    Curves are thinned to ~60 points so the metrics JSON stays small.
    """
    curves = []
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
        a = float(auc(fpr, tpr))
        idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
        curves.append({
            "label": name,
            "auc": a,
            "fpr": [round(float(v), 4) for v in fpr[idx]],
            "tpr": [round(float(v), 4) for v in tpr[idx]],
        })
    macro = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    return curves, macro


def _calibration(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability data for the model's confidence (max-probability prediction).

    For each confidence bin we report mean predicted confidence vs observed
    accuracy (the ideal calibrated model lies on the diagonal). We also return
    the Expected Calibration Error (ECE).
    """
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece, n = [], 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.sum() == 0:
            continue
        bin_conf = float(conf[m].mean())
        bin_acc = float(correct[m].mean())
        rows.append({
            "confidence": round(bin_conf, 4),
            "accuracy": round(bin_acc, 4),
            "count": int(m.sum()),
        })
        ece += (m.sum() / n) * abs(bin_acc - bin_conf)
    return [{"ece": round(float(ece), 4)}] + rows
