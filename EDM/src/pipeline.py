"""
Data pipeline: turn a raw OpenStreetMap road network into the exact feature
frame the congestion model expects.

This module is a clean, reusable refactor of the exploratory work done in the
internship notebooks (``01-limpieza_orientativa`` and ``02-conjunto_datos_espiras``).
The key idea that makes the application original is *transfer*: we learn priors
(median lanes / speed per road class) on a sensor-rich city (Valencia) and reuse
them to clean and score a town that has **no traffic sensors at all** (Paiporta).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Feature contract (must stay identical between training and inference)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES: list[str] = ["lanes", "maxspeed_final", "length"]
BINARY_FEATURES: list[str] = [
    "lanes_missing",
    "maxspeed_missing",
    "oneway",
    "reversed",
    "es_tunel",
    "es_puente",
    "es_rotonda",
    "acceso_restringido",
]

# Road classes seen in Valencia (drives the one-hot columns). Frozen here so the
# model always receives the same columns, even when Paiporta lacks some classes.
HIGHWAY_CATEGORIES: list[str] = [
    "residential", "unclassified", "primary", "tertiary", "secondary",
    "living_street", "trunk_link", "trunk", "primary_link", "tertiary_link",
    "secondary_link", "motorway_link", "motorway", "busway",
]

# Ordinal target: low < medium < high < peak congestion.
CLASS_NAMES: list[str] = ["low", "medium", "high", "peak"]
CLASS_NAMES_ES: list[str] = ["bajo", "medio", "alto", "pico"]


# --------------------------------------------------------------------------- #
# Priors transferred from a sensor-rich city to a sensorless one
# --------------------------------------------------------------------------- #
@dataclass
class CityPriors:
    """Per-road-class medians used to impute missing lanes / max-speed."""

    lanes_by_highway: dict[str, float] = field(default_factory=dict)
    maxspeed_by_highway: dict[str, float] = field(default_factory=dict)
    lanes_global: float = 1.0
    maxspeed_global: float = 50.0

    def to_dict(self) -> dict:
        return {
            "lanes_by_highway": self.lanes_by_highway,
            "maxspeed_by_highway": self.maxspeed_by_highway,
            "lanes_global": self.lanes_global,
            "maxspeed_global": self.maxspeed_global,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CityPriors":
        return cls(**d)


# --------------------------------------------------------------------------- #
# Parsing helpers (OSM fields are messy: lists, ranges, units, strings)
# --------------------------------------------------------------------------- #
def _first_value(value):
    """OSM tags sometimes arrive as a Python-list-like string or a real list."""
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) else None
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and parsed:
                return parsed[0]
        except (ValueError, SyntaxError):
            pass
    return value


def parse_numeric(value) -> float:
    """Extract the first plausible number from messy OSM tags ('50 km/h', '2;3')."""
    value = _first_value(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def _is_truthy_tag(value, positives: set[str]) -> int:
    value = _first_value(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(str(value).lower() in positives)


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Return a column as a Series, or a default-filled Series if it is absent.

    OSMnx omits tags that no edge carries (e.g. a small town with no tunnels),
    so we must never assume a column exists.
    """
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_osm_edges(edges: pd.DataFrame, priors: CityPriors | None = None) -> pd.DataFrame:
    """
    Reproduce the notebook cleaning on any OSMnx ``edges`` GeoDataFrame and
    return a frame carrying geometry + every model input.

    When ``priors`` is supplied, missing ``lanes`` / ``maxspeed`` are imputed
    with those transferred medians (so a sensorless town inherits Valencia's
    domain knowledge); otherwise medians are computed in-place.
    """
    df = edges.reset_index(drop=True).copy()

    # Collapse list-valued tags to a single representative value.
    for col in ["highway", "junction", "tunnel", "bridge", "access", "name", "oneway", "reversed"]:
        if col in df.columns:
            df[col] = df[col].apply(_first_value)

    df["highway"] = df["highway"].astype("object").where(df["highway"].notna(), "unclassified")

    # Geometry endpoints (used by the map and the sensor-matching step).
    df["lon_A"] = df.geometry.apply(lambda g: g.coords[0][0] if g is not None else np.nan)
    df["lat_A"] = df.geometry.apply(lambda g: g.coords[0][1] if g is not None else np.nan)
    df["lon_B"] = df.geometry.apply(lambda g: g.coords[-1][0] if g is not None else np.nan)
    df["lat_B"] = df.geometry.apply(lambda g: g.coords[-1][1] if g is not None else np.nan)

    # Numeric raw values.
    df["lanes"] = _col(df, "lanes").apply(parse_numeric)
    df["maxspeed"] = _col(df, "maxspeed").apply(parse_numeric)

    # Missingness indicators (informative on their own).
    df["lanes_missing"] = df["lanes"].isna().astype(int)
    df["maxspeed_missing"] = df["maxspeed"].isna().astype(int)

    # Structural binary features.
    df["es_tunel"] = _col(df, "tunnel").apply(lambda v: _is_truthy_tag(v, {"yes"}))
    df["es_puente"] = _col(df, "bridge").apply(lambda v: _is_truthy_tag(v, {"yes"}))
    df["es_rotonda"] = _col(df, "junction").apply(
        lambda v: _is_truthy_tag(v, {"roundabout", "circular"})
    )
    df["acceso_restringido"] = _col(df, "access").apply(
        lambda v: _is_truthy_tag(v, {"no", "private", "destination", "permit"})
    )

    # Direction flags to int.
    df["oneway"] = _col(df, "oneway", False).fillna(False).astype(bool).astype(int)
    df["reversed"] = _col(df, "reversed", False).fillna(False).astype(bool).astype(int)

    if priors is None:
        priors = compute_priors(df)

    # Impute lanes / max-speed using (transferred) per-class medians.
    df["lanes"] = _impute_by_highway(df, "lanes", priors.lanes_by_highway, priors.lanes_global)
    df["maxspeed_final"] = _impute_by_highway(
        df, "maxspeed", priors.maxspeed_by_highway, priors.maxspeed_global
    )

    if "length" not in df.columns:
        df["length"] = df.geometry.length  # degrees; only a fallback
    df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0.0)

    if "id_tramo" not in df.columns:
        df["id_tramo"] = np.arange(1, len(df) + 1)
    df["name"] = _col(df, "name", "").fillna("").astype(str)
    return df


def _impute_by_highway(df, col, by_highway, global_value):
    filled = df[col].copy()
    mask = filled.isna()
    if mask.any():
        filled.loc[mask] = df.loc[mask, "highway"].map(by_highway)
    return pd.to_numeric(filled, errors="coerce").fillna(global_value)


def compute_priors(clean_or_raw: pd.DataFrame) -> CityPriors:
    """Learn per-road-class medians from a (cleaned) frame."""
    df = clean_or_raw
    lanes = pd.to_numeric(df["lanes"], errors="coerce")
    speed_col = "maxspeed_final" if "maxspeed_final" in df.columns else "maxspeed"
    speed = pd.to_numeric(df[speed_col], errors="coerce")
    return CityPriors(
        lanes_by_highway=lanes.groupby(df["highway"]).median().dropna().to_dict(),
        maxspeed_by_highway=speed.groupby(df["highway"]).median().dropna().to_dict(),
        lanes_global=float(np.nanmedian(lanes)) if lanes.notna().any() else 1.0,
        maxspeed_global=float(np.nanmedian(speed)) if speed.notna().any() else 50.0,
    )


# --------------------------------------------------------------------------- #
# Feature frame (identical columns for train + every inference target)
# --------------------------------------------------------------------------- #
def build_feature_frame(df: pd.DataFrame, highway_categories: list[str] | None = None) -> pd.DataFrame:
    """Return the model design matrix with a frozen, ordered column set."""
    cats = highway_categories or HIGHWAY_CATEGORIES
    out = pd.DataFrame(index=df.index)

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BINARY_FEATURES:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    hw = df["highway"].astype("object")
    for cat in cats:
        out[f"highway_{cat}"] = (hw == cat).astype(int)

    return out


def feature_columns(highway_categories: list[str] | None = None) -> list[str]:
    cats = highway_categories or HIGHWAY_CATEGORIES
    return NUMERIC_FEATURES + BINARY_FEATURES + [f"highway_{c}" for c in cats]
