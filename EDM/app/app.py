"""
Sensorless Traffic — street-level congestion estimates for an entire city,
including the streets that have no sensors.

EDM course project · València open data + OpenStreetMap · XGBoost · Streamlit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

EDM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDM_DIR))

from src.io_live import LiveDataError, fetch_traffic_state  # noqa: E402
from src.modeling import CongestionModel  # noqa: E402
from src.pipeline import CLASS_NAMES, HIGHWAY_CATEGORIES  # noqa: E402

ART = EDM_DIR / "artifacts"
DATA = EDM_DIR / "data"

# Institutional palette — primary teal is used for chrome; the green→red scale
# is reserved strictly for congestion data (no colour collisions).
PRIMARY = "#0F6E82"
INK = "#16323F"
MUTED = "#5B6B73"
LINE = "#E2E9EC"

LEVELS = ["low", "medium", "high", "peak"]
LABELS = {"low": "Low", "medium": "Medium", "high": "High", "peak": "Peak"}
HEX = {"low": "#1FA45F", "medium": "#E8B210", "high": "#E2742B", "peak": "#C0392B"}
GREY = "#C7D0D6"
LEVEL_HELP = {
    "low": "Free-flowing",
    "medium": "Steady traffic",
    "high": "Busy, near saturation",
    "peak": "Arterial-level demand",
}

st.set_page_config(
    page_title="Sensorless Traffic · València",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}}
      #MainMenu, footer {{visibility: hidden;}}

      /* Header band */
      .appbar {{border-bottom: 2px solid {PRIMARY}; padding: 0 0 0.9rem 0; margin-bottom: 1.1rem;}}
      .appbar .kicker {{font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
            color: {PRIMARY}; text-transform: uppercase;}}
      .appbar .title {{font-size: 2.05rem; font-weight: 800; color: {INK};
            line-height: 1.1; margin: 0.15rem 0 0.25rem 0; letter-spacing: -0.02em;}}
      .appbar .sub {{font-size: 1.0rem; color: {MUTED}; max-width: 70ch;}}

      /* KPI cards */
      .kpi {{background: #fff; border: 1px solid {LINE}; border-left: 3px solid {PRIMARY};
            border-radius: 10px; padding: 0.85rem 1.05rem; height: 100%;}}
      .kpi .k-label {{font-size: 0.74rem; font-weight: 600; letter-spacing: 0.04em;
            color: {MUTED}; text-transform: uppercase;}}
      .kpi .k-value {{font-size: 1.85rem; font-weight: 800; color: {INK}; line-height: 1.15;
            margin-top: 0.1rem;}}
      .kpi .k-sub {{font-size: 0.82rem; color: {MUTED}; margin-top: 0.05rem;}}

      /* Sidebar */
      .brand {{font-size: 1.12rem; font-weight: 800; color: {INK};}}
      .brand-sub {{font-size: 0.8rem; color: {MUTED}; margin-bottom: 0.2rem;}}
      .leg {{display:flex; align-items:center; gap:9px; margin: 5px 0; font-size: 0.86rem;}}
      .leg .sw {{width: 16px; height: 10px; border-radius: 2px; flex: 0 0 auto;}}
      .leg .lv {{font-weight: 700; color: {INK}; min-width: 56px;}}
      .leg .rg {{color: {MUTED}; font-size: 0.78rem;}}
      .src {{font-size: 0.78rem; color: #8A969C;}}

      /* Tabs */
      button[data-baseweb="tab"] {{font-size: 0.96rem; font-weight: 600;}}
      div[data-baseweb="tab-list"] {{gap: 6px; border-bottom: 1px solid {LINE};}}

      /* Chips */
      .chip {{display:inline-block; padding:2px 9px; border-radius:6px; font-size:0.74rem;
            font-weight:700; color:#fff; margin-right:5px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model() -> CongestionModel:
    return CongestionModel.load(ART / "model.joblib")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


@st.cache_data(show_spinner="Loading the road network…")
def load_segments(name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / name)
    df["path"] = df["path"].apply(lambda p: [[float(a), float(b)] for a, b in p])
    df["name"] = df["name"].fillna("").astype(str)
    return df


@st.cache_data(ttl=180, show_spinner=False)
def load_live():
    """Return (segments, is_live). Falls back to the bundled snapshot offline."""
    try:
        return fetch_traffic_state(), True
    except LiveDataError:
        snap = json.loads((DATA / "measured_snapshot.json").read_text())
        return snap, False


MODEL = load_model()
METRICS = load_metrics()
THRESH = [max(b, 0) for b in MODEL.qcut_bins]


def level_range(level: str) -> str:
    i = LEVELS.index(level)
    return f"{THRESH[i]:.0f}–{THRESH[i + 1]:.0f} veh/h"


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def kpi(col, label, value, sub="", accent=PRIMARY):
    col.markdown(
        f'<div class="kpi" style="border-left-color:{accent}">'
        f'<div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>'
        f'<div class="k-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def legend(levels=LEVELS, ranges=True):
    rows = []
    for l in levels:
        rg = f'<span class="rg">{level_range(l)}</span>' if ranges else ""
        rows.append(
            f'<div class="leg"><span class="sw" style="background:{HEX[l]}"></span>'
            f'<span class="lv">{LABELS[l]}</span>{rg}</div>'
        )
    return "".join(rows)


def _geojson(df: pd.DataFrame) -> dict:
    """Build a LineString FeatureCollection from a frame carrying path/color_hex/name/tip_level."""
    feats = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": r.path},
            "properties": {"color": r.color_hex, "name": r.name or "Unnamed street",
                           "level": r.tip_level},
        }
        for r in df.itertuples()
    ]
    return {"type": "FeatureCollection", "features": feats}


def _legend_element(items) -> folium.Element:
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:15px;height:9px;'
        f'background:{c};border-radius:2px;margin-right:7px;vertical-align:middle"></span>{lbl}</div>'
        for c, lbl in items
    )
    html = (
        '<div style="position:fixed;bottom:22px;right:14px;z-index:9999;'
        'background:rgba(255,255,255,.94);border:1px solid #E2E9EC;border-radius:8px;'
        'padding:9px 12px;font-family:sans-serif;font-size:12px;font-weight:600;color:#16323F;'
        'box-shadow:0 2px 7px rgba(0,0,0,.13)">'
        '<div style="font-size:10.5px;color:#5B6B73;text-transform:uppercase;'
        f'letter-spacing:.05em;margin-bottom:4px">Congestion</div>{rows}</div>'
    )
    return folium.Element(html)


def render_map(layers, center, zoom, height=560, legend_items=None):
    """Render one or more network layers on a clean CartoDB-Positron Leaflet map.

    ``layers`` is a list of (df, weight, with_tooltip). Folium + raster tiles is
    used (rather than a vector base map) for reliable rendering on any network.
    A self-contained legend is drawn on the map itself (data-label proximity).
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron",
                   prefer_canvas=True, control_scale=True, zoom_control=True)
    for df, weight, with_tooltip in layers:
        if df.empty:
            continue
        gj = folium.GeoJson(
            _geojson(df),
            style_function=lambda f, w=weight: {
                "color": f["properties"]["color"], "weight": w, "opacity": 0.9},
            highlight_function=lambda f, w=weight: {"weight": w + 3, "opacity": 1},
        )
        if with_tooltip:
            folium.GeoJsonTooltip(fields=["name", "level"],
                                  aliases=["Street", "Congestion"],
                                  sticky=True, labels=True).add_to(gj)
        gj.add_to(m)
    if legend_items:
        m.get_root().html.add_child(_legend_element(legend_items))
    components.html(m.get_root().render(), height=height)


def level_legend(extra=None):
    items = [(HEX[l], LABELS[l]) for l in LEVELS]
    return items + (extra or [])


def color_by(df: pd.DataFrame, level_col: str) -> pd.DataFrame:
    out = df.copy()
    out["color_hex"] = out[level_col].map(HEX)
    out["tip_level"] = out[level_col].map(LABELS)
    return out


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="brand">Sensorless Traffic</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Urban mobility analytics · València open data</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        f'<span style="color:{MUTED};font-size:0.9rem">A planning aid that estimates '
        "congestion on every street — not just the few that carry a sensor — and "
        "extends the estimate to towns that have none.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="brand-sub" style="font-weight:700;color:'
                f'{INK}">Congestion scale</div>', unsafe_allow_html=True)
    st.markdown(legend(), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<span class="src">Sources · '
        '<a href="https://valencia.opendatasoft.com/" target="_blank">València Open Data</a> '
        "(loop sensors + real-time traffic) · OpenStreetMap.<br>Model · gradient-boosted "
        "trees (XGBoost).</span>",
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="appbar">'
    '<div class="kicker">València · Urban mobility analytics</div>'
    '<div class="title">Sensorless Traffic</div>'
    '<div class="sub">Street-level congestion for the whole city — including the '
    "<b>97.6%</b> of streets that have no sensor — plus transfer to a town with none.</div>"
    "</div>",
    unsafe_allow_html=True,
)

tab_map, tab_sim, tab_transfer, tab_model = st.tabs(
    ["Congestion map", "Scenario simulator", "Transfer to Paiporta", "Model & validation"]
)


# =========================================================================== #
# TAB 1 — Congestion map (Valencia)
# =========================================================================== #
with tab_map:
    val = load_segments("valencia_segments.parquet")
    n_total = len(val)
    n_meas = int(val["measured_level"].notna().sum())

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Street segments", f"{n_total:,}", "Valencia road network (OSM)")
    kpi(c2, "Covered by a sensor", f"{n_meas:,}", f"{n_meas / n_total:.1%} of the network",
        accent=HEX["high"])
    kpi(c3, "Estimated by the model", f"{n_total - n_meas:,}",
        f"{(n_total - n_meas) / n_total:.1%} of the network", accent=HEX["low"])
    kpi(c4, "Mean confidence", f"{val['pred_conf'].mean():.0%}", "Across predicted segments")

    st.write("")
    left, right = st.columns([1, 3.1])
    with left:
        mode = st.radio(
            "Layer",
            ["Model estimate (whole city)", "Sensor coverage only"],
            help="‘Sensor coverage only’ shows how little of the city is actually measured.",
        )
        classes = st.multiselect(
            "Road classes",
            HIGHWAY_CATEGORIES,
            default=["primary", "secondary", "tertiary", "trunk", "motorway"],
            help="Residential streets are ~half of all segments — add them to see the full network.",
        )
        shown = st.multiselect("Congestion levels", LEVELS, default=LEVELS,
                               format_func=lambda l: LABELS[l])
        if "residential" in classes:
            st.caption("Large view — residential streets included may take a moment.")

    base = val[val["highway"].isin(classes)] if classes else val

    if mode.startswith("Sensor"):
        measured = color_by(base[base["measured_level"].isin(shown)], "measured_level")
        blind = base[base["measured_level"].isna()].copy()
        blind["color_hex"] = GREY
        blind["tip_level"] = "No sensor"
        layers = [(blind, 1.3, False), (measured, 4.5, True)]
        legend_items = level_legend([(GREY, "No sensor")])
        caption = (f"**{len(measured):,}** measured segments in colour · "
                   f"**{len(blind):,}** unmeasured (grey). This is the city's current visibility.")
    else:
        view = color_by(base[base["pred_level"].isin(shown)], "pred_level")
        layers = [(view, 2.2, True)]
        legend_items = level_legend()
        caption = f"Model-estimated congestion across **{len(view):,}** segments."

    with right:
        with st.spinner("Rendering the network…"):
            render_map(layers, [39.466, -0.376], 12, height=560, legend_items=legend_items)
        st.caption(caption)


# =========================================================================== #
# TAB 2 — Scenario simulator
# =========================================================================== #
with tab_sim:
    st.markdown("#### Redesign a street, see the impact")
    st.markdown(
        f'<span style="color:{MUTED}">Choose a real Valencia street (or design one), '
        "change its physical attributes, and the model re-estimates its congestion "
        "level instantly — a what-if tool for use <i>before</i> any works begin.</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    val = load_segments("valencia_segments.parquet")
    named = val[val["name"].str.len() > 0].drop_duplicates("name").sort_values("name")
    options = ["— Design a street from scratch —"] + named["name"].tolist()
    pick = st.selectbox("Street template", options, index=0)

    if pick == options[0]:
        template = {"highway": "secondary", "lanes": 2.0, "maxspeed_final": 50.0, "length": 200.0,
                    "oneway": 0, "reversed": 0, "lanes_missing": 0, "maxspeed_missing": 0,
                    "es_tunel": 0, "es_puente": 0, "es_rotonda": 0, "acceso_restringido": 0}
    else:
        template = named[named["name"] == pick].iloc[0].to_dict()

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("**Street attributes**")
        hw = st.selectbox("Road class", HIGHWAY_CATEGORIES,
                          index=HIGHWAY_CATEGORIES.index(template["highway"])
                          if template["highway"] in HIGHWAY_CATEGORIES else 0)
        lanes = st.slider("Lanes", 1, 6, int(np.clip(template["lanes"], 1, 6)))
        speeds = [20, 30, 40, 50, 60, 70, 80, 100, 120]
        speed = st.select_slider("Speed limit (km/h)", speeds,
                                 value=min(speeds, key=lambda v: abs(v - template["maxspeed_final"])))
        length = st.slider("Segment length (m)", 10, 1000, int(np.clip(template["length"], 10, 1000)))
        oneway = st.checkbox("One-way", value=bool(template["oneway"]))
        roundabout = st.checkbox("Roundabout", value=bool(template["es_rotonda"]))

    row = {"highway": hw, "lanes": float(lanes), "maxspeed_final": float(speed),
           "length": float(length), "oneway": int(oneway), "reversed": 0,
           "lanes_missing": 0, "maxspeed_missing": 0, "es_tunel": int(template["es_tunel"]),
           "es_puente": int(template["es_puente"]), "es_rotonda": int(roundabout),
           "acceso_restringido": int(template["acceso_restringido"])}
    proba = MODEL.predict_proba(pd.DataFrame([row])).iloc[0]
    pred = LEVELS[int(np.argmax(proba.to_numpy()))]

    with colB:
        st.markdown("**Estimated congestion**")
        st.markdown(
            f'<div style="font-size:2.3rem;font-weight:800;color:{HEX[pred]};line-height:1.1">'
            f'{LABELS[pred]}</div>'
            f'<div class="src">{level_range(pred)} · {LEVEL_HELP[pred]}</div>',
            unsafe_allow_html=True,
        )
        chart_df = pd.DataFrame({"level": [LABELS[l] for l in LEVELS],
                                 "probability": [proba[l] for l in LEVELS]})
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadius=3)
            .encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%"), title="Model probability"),
                y=alt.Y("level:N", sort=[LABELS[l] for l in LEVELS], title=None),
                color=alt.Color("level:N",
                                scale=alt.Scale(domain=[LABELS[l] for l in LEVELS],
                                                range=[HEX[l] for l in LEVELS]),
                                legend=None),
            )
            .properties(height=190)
        )
        st.altair_chart(chart, width="stretch")
        if pick != options[0]:
            st.caption(f"Current model estimate for **{pick}**: "
                       f"**{LABELS.get(template.get('pred_level'), '—')}** → "
                       f"your redesign: **{LABELS[pred]}**.")
    st.info(
        "Take a quiet residential street, reclassify it as primary with 4 lanes at "
        "60 km/h — the model shifts it toward High/Peak. That is exactly the trade-off "
        "a mobility plan has to weigh, here without installing a single sensor."
    )


# =========================================================================== #
# TAB 3 — Transfer to Paiporta
# =========================================================================== #
with tab_transfer:
    st.markdown("#### Scoring a town with zero sensors")
    st.markdown(
        f'<span style="color:{MUTED}">Paiporta, adjacent to València, has no traffic '
        "loop sensors. We apply the València-trained model and its learned per-road-class "
        "priors to produce a full congestion map for Paiporta, built entirely from "
        "OpenStreetMap geometry.</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    try:
        pai = load_segments("paiporta_segments.parquet")
        k1, k2, k3 = st.columns(3)
        kpi(k1, "Paiporta segments", f"{len(pai):,}", "From OpenStreetMap")
        kpi(k2, "Local sensors used", "0", "None exist", accent=HEX["peak"])
        kpi(k3, "Coverage from transfer", "100%", "Every street estimated", accent=HEX["low"])

        st.write("")
        left, right = st.columns([1, 3.1])
        with left:
            lv = st.multiselect("Congestion levels", LEVELS, default=LEVELS,
                                format_func=lambda l: LABELS[l], key="pai_levels")
            dist = pai["pred_level"].value_counts().reindex(LEVELS).fillna(0).astype(int)
            st.markdown('<div class="brand-sub" style="margin-top:6px;font-weight:700;'
                        f'color:{INK}">Estimated mix</div>', unsafe_allow_html=True)
            for l in LEVELS:
                st.markdown(
                    f'<div class="leg"><span class="sw" style="background:{HEX[l]}"></span>'
                    f'<span class="lv">{LABELS[l]}</span>'
                    f'<span class="rg">{dist[l]} segments</span></div>',
                    unsafe_allow_html=True,
                )
        pv = color_by(pai[pai["pred_level"].isin(lv)], "pred_level")
        with right:
            with st.spinner("Rendering Paiporta…"):
                render_map([(pv, 3.2, True)], [39.4275, -0.4185], 14, height=520,
                           legend_items=level_legend())
            st.caption("Every street here was estimated with **no local ground truth** — "
                       "a spatial transfer of the València model.")
    except FileNotFoundError:
        st.warning("Paiporta artifact not found. Run `EDM/scripts/build_artifacts.py`.")


# =========================================================================== #
# TAB 4 — Model & validation
# =========================================================================== #
with tab_model:
    st.markdown("#### How it works, and how well")

    m1, m2, m3, m4, m5 = st.columns(5)
    kpi(m1, "Test F1-macro", f"{METRICS['test_f1_macro']:.3f}", "Stratified hold-out")
    kpi(m2, "Balanced acc.", f"{METRICS['test_balanced_accuracy']:.3f}", "4 classes")
    kpi(m3, "Cohen's κ", f"{METRICS['test_kappa']:.3f}", "Agreement vs chance")
    kpi(m4, "Macro AUC", f"{METRICS.get('test_macro_auc', float('nan')):.3f}",
        "One-vs-rest ROC")
    kpi(m5, "5-fold CV F1", f"{METRICS['cv_f1_macro_mean']:.3f}",
        f"± {METRICS['cv_f1_macro_std']:.2f} across folds")
    st.caption(
        f"Trained on **{METRICS['n_labeled']:,}** sensor-labelled segments "
        f"({METRICS['n_train']:,} train / {METRICS['n_test']:,} hold-out test). A "
        "majority-class baseline scores ≈0.10 F1-macro, so the model roughly "
        "quintuples it from static road attributes alone."
    )

    st.write("")
    cL, cR = st.columns(2)
    with cL:
        st.markdown("**What drives the estimate** — feature importance")
        fi = pd.DataFrame(METRICS["feature_importance"]).head(12)
        fi["feature"] = fi["feature"].str.replace("highway_", "class · ").str.replace("_", " ")
        chart = (
            alt.Chart(fi)
            .mark_bar(color=PRIMARY, cornerRadius=2)
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Road class and lane count dominate — congestion is largely written "
                   "into a street's physical hierarchy.")

    with cR:
        st.markdown("**Confusion matrix** — hold-out test")
        cm = np.array(METRICS["confusion_matrix"])
        cm_df = pd.DataFrame(
            [(LABELS[CLASS_NAMES[i]], LABELS[CLASS_NAMES[j]], int(cm[i, j]))
             for i in range(len(CLASS_NAMES)) for j in range(len(CLASS_NAMES))],
            columns=["actual", "predicted", "count"],
        )
        order = [LABELS[c] for c in CLASS_NAMES]
        heat = (
            alt.Chart(cm_df)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", sort=order, title="Predicted"),
                y=alt.Y("actual:N", sort=order, title="Actual"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="teals"), legend=None),
            )
            .properties(height=300)
        )
        text = heat.mark_text(baseline="middle", fontSize=14).encode(
            text="count:Q",
            color=alt.condition("datum.count > 40", alt.value("white"), alt.value(INK)),
        )
        st.altair_chart(heat + text, width="stretch")
        st.caption("Errors fall mostly between adjacent levels — the model rarely "
                   "mistakes a quiet street for an artery.")

    if METRICS.get("roc") and METRICS.get("calibration"):
        st.write("")
        rL, rR = st.columns(2)
        with rL:
            st.markdown("**ROC curves** — one-vs-rest, per class")
            roc_rows = []
            for c in METRICS["roc"]:
                lbl = f"{LABELS[c['label']]} (AUC {c['auc']:.2f})"
                for fpr, tpr in zip(c["fpr"], c["tpr"]):
                    roc_rows.append({"fpr": fpr, "tpr": tpr, "class": lbl})
            roc_df = pd.DataFrame(roc_rows)
            domain = sorted(roc_df["class"].unique())
            rng = [HEX[c["label"]] for c in METRICS["roc"]]
            roc_chart = (
                alt.Chart(roc_df)
                .mark_line(strokeWidth=2.4)
                .encode(
                    x=alt.X("fpr:Q", title="False-positive rate",
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("tpr:Q", title="True-positive rate",
                            scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("class:N", scale=alt.Scale(domain=domain, range=rng),
                                    legend=alt.Legend(title=None, orient="bottom-right")),
                )
                .properties(height=300)
            )
            diag = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(strokeDash=[4, 4], color="#9AA7AD")
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart(diag + roc_chart, width="stretch")
            st.caption("Peak congestion is the most separable (AUC≈0.87); the *High* "
                       "band is the hardest — the classes either side bleed into it.")

        with rR:
            ece = METRICS["calibration"][0]["ece"]
            st.markdown(f"**Calibration (reliability)** — ECE = {ece:.3f}")
            cal_df = pd.DataFrame(METRICS["calibration"][1:])
            pts = (
                alt.Chart(cal_df)
                .mark_line(point=True, strokeWidth=2.4, color=PRIMARY)
                .encode(
                    x=alt.X("confidence:Q", title="Predicted confidence",
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("accuracy:Q", title="Observed accuracy",
                            scale=alt.Scale(domain=[0, 1])),
                    tooltip=[alt.Tooltip("confidence:Q", format=".2f"),
                             alt.Tooltip("accuracy:Q", format=".2f"),
                             alt.Tooltip("count:Q", title="segments")],
                )
                .properties(height=300)
            )
            diag2 = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(strokeDash=[4, 4], color="#9AA7AD")
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart(diag2 + pts, width="stretch")
            st.caption("Points near the diagonal mean the model's confidence is "
                       "trustworthy — a low ECE says its probabilities are well calibrated.")

    st.write("")
    st.markdown("**Methodology — an end-to-end data-science pipeline**")
    steps = [
        ("1 · Acquire", "València road graph from OpenStreetMap (OSMnx) and hourly "
                        "counts from the city's loop sensors."),
        ("2 · Clean & impute", "Per-road-class median imputation for lanes; spatial "
                               "k-NN imputation for speed limits; missingness kept as a feature."),
        ("3 · Spatial join", "A Haversine point-on-segment algorithm assigns each "
                            "sensor to its street segment."),
        ("4 · Model", "Quartile congestion target; baseline → tree → random forest → "
                     "MLP → XGBoost, with cross-validation and ANOVA / Tukey checks."),
    ]
    cols = st.columns(4)
    for col, (h, b) in zip(cols, steps):
        col.markdown(f'<div class="kpi" style="border-left-color:{PRIMARY}">'
                     f'<div class="k-label">{h}</div>'
                     f'<div class="k-sub" style="margin-top:6px">{b}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    with st.expander("Live data feed & limitations"):
        seg, is_live = load_live()
        if is_live:
            st.success(f"Live: {len(seg):,} real-time traffic segments pulled from "
                       "València Open Data (refreshes every ~3 min).")
        else:
            st.info("Live portal not reachable from this network — showing the bundled "
                    "measured snapshot. The public deployment uses the real-time feed.")
        st.markdown(
            "- The model learns from a **single one-hour snapshot**, so it captures the "
            "*spatial* structure of congestion, not its time-of-day dynamics. Historical "
            "and real-time series are the natural next step.\n"
            "- Transfer to Paiporta assumes its streets behave like València streets of the "
            "same class — reasonable for a neighbouring town, to be validated once sensors exist.\n"
            "- Metrics are reported honestly: κ ≈ 0.39 reflects a genuinely hard, data-scarce "
            "problem (1,292 labelled segments), not an over-tuned demo."
        )

st.markdown(
    f'<div style="border-top:1px solid {LINE};margin-top:1.6rem;padding-top:0.7rem">'
    f'<span class="src">Open-data prototype · built on '
    '<a href="https://valencia.opendatasoft.com/" target="_blank">València Open Data</a> '
    "and OpenStreetMap · not an official service of the Ajuntament de València. "
    "EDM course project.</span></div>",
    unsafe_allow_html=True,
)
