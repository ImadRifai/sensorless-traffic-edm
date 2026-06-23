"""
Build every artifact the Streamlit app consumes, from the raw internship data.

Outputs (under EDM/artifacts and EDM/data):
  - artifacts/model.joblib      trained CongestionModel bundle
  - artifacts/metrics.json      honest held-out + CV metrics, feature importance
  - data/valencia_segments.parquet   full network scored (measured + predicted)
  - data/paiporta_segments.parquet   transfer: a sensorless town scored
  - data/measured_snapshot.json      offline fallback for the live tab

Run from the repo root:  .venv/bin/python EDM/scripts/build_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "EDM"))

from src.modeling import CongestionModel, train_and_evaluate  # noqa: E402
from src.pipeline import (  # noqa: E402
    CLASS_NAMES,
    CityPriors,
    build_feature_frame,
    clean_osm_edges,
    compute_priors,
)

DATA_IN = ROOT / "data"
ART = ROOT / "EDM" / "artifacts"
DATA_OUT = ROOT / "EDM" / "data"
ART.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

MODEL_INPUTS = [
    "highway", "lanes", "maxspeed_final", "length", "oneway", "reversed",
    "lanes_missing", "maxspeed_missing", "es_tunel", "es_puente",
    "es_rotonda", "acceso_restringido",
]


def _path_from_geom(geom, ndigits=5):
    if geom is None or geom.is_empty:
        return None
    coords = list(geom.coords)
    return [[round(x, ndigits), round(y, ndigits)] for x, y in coords]


def _measured_index(vph: pd.Series, bins: list[float]) -> pd.Series:
    """Bucket measured vehicles/hour into the model's quartile classes."""
    edges = np.array(bins, dtype=float)
    idx = np.digitize(vph.to_numpy(dtype=float), edges[1:-1], right=True)
    out = pd.Series(idx, index=vph.index, dtype="float")
    out[vph.isna()] = np.nan
    return out


def build_segment_table(clean_df: pd.DataFrame, model: CongestionModel) -> pd.DataFrame:
    df = clean_df.copy()
    df["highway"] = df["highway"].fillna("unclassified")
    df["pred_idx"] = np.argmax(model.predict_proba(df).to_numpy(), axis=1)
    df["pred_level"] = np.asarray(CLASS_NAMES)[df["pred_idx"].astype(int)]
    df["pred_conf"] = model.predict_proba(df).max(axis=1).to_numpy()
    return df


def main() -> None:
    # ---------------------------------------------------------------- #
    # 1. Train + evaluate on Valencia's sensor-labeled segments
    # ---------------------------------------------------------------- #
    print("[1/5] Loading Valencia model dataset...")
    df = pd.read_csv(DATA_IN / "datos_modelo.csv")
    df["highway"] = df["highway"].fillna("unclassified")

    print("[2/5] Training + evaluating XGBoost...")
    result = train_and_evaluate(df, target_col="vehiculos_por_hora")
    model: CongestionModel = result["model"]
    metrics = result["metrics"]

    priors = compute_priors(df)
    model.priors = priors.to_dict()
    model.save(ART / "model.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"      test F1-macro={metrics['test_f1_macro']:.3f} "
          f"kappa={metrics['test_kappa']:.3f} "
          f"CV F1={metrics['cv_f1_macro_mean']:.3f}±{metrics['cv_f1_macro_std']:.3f}")

    # ---------------------------------------------------------------- #
    # 2. Score the WHOLE Valencia network (measured + predicted)
    # ---------------------------------------------------------------- #
    print("[3/5] Scoring the full Valencia network...")
    geom = df["geometry"].apply(wkt.loads)
    scored = build_segment_table(df, model)
    scored["path"] = geom.apply(_path_from_geom)
    scored["measured_idx"] = _measured_index(
        pd.to_numeric(df["vehiculos_por_hora"], errors="coerce"), model.qcut_bins
    )
    scored["measured_level"] = scored["measured_idx"].apply(
        lambda i: CLASS_NAMES[int(i)] if pd.notna(i) else None
    )
    scored["vehiculos_por_hora"] = pd.to_numeric(df["vehiculos_por_hora"], errors="coerce")
    scored["name"] = df["name"].fillna("")

    cols = ["id_tramo", "name"] + MODEL_INPUTS + [
        "pred_idx", "pred_level", "pred_conf",
        "measured_idx", "measured_level", "vehiculos_por_hora", "path",
    ]
    valencia = scored[[c for c in cols if c in scored.columns]].dropna(subset=["path"])
    valencia.to_parquet(DATA_OUT / "valencia_segments.parquet", index=False)
    n_meas = valencia["measured_level"].notna().sum()
    print(f"      {len(valencia):,} segments | {n_meas:,} measured "
          f"({n_meas / len(valencia):.1%}) | {len(valencia) - n_meas:,} predicted only")

    # Offline fallback snapshot for the live tab (measured segments only).
    snap = valencia[valencia["measured_level"].notna()][
        ["name", "measured_level", "vehiculos_por_hora", "path"]
    ]
    (DATA_OUT / "measured_snapshot.json").write_text(
        snap.to_json(orient="records"), encoding="utf-8"
    )

    # ---------------------------------------------------------------- #
    # 3. TRANSFER: score Paiporta (a town with NO sensors)
    # ---------------------------------------------------------------- #
    print("[4/5] Downloading + scoring Paiporta by transfer...")
    try:
        import osmnx as ox

        ox.settings.use_cache = True
        ox.settings.requests_timeout = 90
        g = ox.graph_from_place("Paiporta, Valencia, Spain", network_type="drive", simplify=True)
        edges = ox.graph_to_gdfs(g, nodes=False, edges=True)
        clean = clean_osm_edges(edges, priors=CityPriors.from_dict(model.priors))
        pai = build_segment_table(clean, model)
        pai["path"] = clean.geometry.apply(_path_from_geom)
        pai["measured_level"] = None
        pai["measured_idx"] = np.nan
        pai["vehiculos_por_hora"] = np.nan
        out_cols = ["id_tramo", "name"] + MODEL_INPUTS + [
            "pred_idx", "pred_level", "pred_conf",
            "measured_idx", "measured_level", "vehiculos_por_hora", "path",
        ]
        pai = pai[[c for c in out_cols if c in pai.columns]].dropna(subset=["path"])
        pai.to_parquet(DATA_OUT / "paiporta_segments.parquet", index=False)
        print(f"      Paiporta: {len(pai):,} segments scored (0 sensors).")
    except Exception as exc:  # noqa: BLE001
        print(f"      [WARN] Paiporta transfer skipped: {exc}")

    print("[5/5] Done. Artifacts written to EDM/artifacts and EDM/data.")


if __name__ == "__main__":
    main()
