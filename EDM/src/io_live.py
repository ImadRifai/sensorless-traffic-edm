"""
Live data access for Valencia's open-data portal.

Primary source is the *real-time traffic state* dataset (updated every ~3 min):
    estat-transit-temps-real-estado-trafico-tiempo-real

We query the Opendatasoft Explore API v2.1 and gracefully degrade: if the portal
is unreachable (e.g. behind a restrictive network), the caller falls back to the
bundled measured snapshot so the app never breaks.
"""
from __future__ import annotations

import requests

# Opendatasoft mirrors — tried in order.
_DOMAINS = [
    "https://valencia.opendatasoft.com",
    "https://valencia.aws-ec2-eu-central-1.opendatasoft.com",
    "https://data.opendatasoft.com",
]
_DATASET = "estat-transit-temps-real-estado-trafico-tiempo-real"

# Official "estado" encoding for the real-time traffic dataset.
TRAFFIC_STATE = {
    0: ("Fluid", "#2ecc71"),
    1: ("Dense", "#f1c40f"),
    2: ("Congested", "#e67e22"),
    3: ("Cut off", "#c0392b"),
    4: ("No data", "#95a5a6"),
    5: ("Maintenance", "#7f8c8d"),
    6: ("No data", "#95a5a6"),
    9: ("No data", "#95a5a6"),
}


class LiveDataError(RuntimeError):
    """Raised when no mirror returns usable live data."""


def _records(domain: str, dataset: str, timeout: int = 12) -> list[dict]:
    suffix = "@valencia" if domain.endswith("data.opendatasoft.com") else ""
    base = f"{domain}/api/explore/v2.1/catalog/datasets/{dataset}{suffix}/records"
    out, offset, limit = [], 0, 100
    while True:
        resp = requests.get(base, params={"limit": limit, "offset": offset}, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        out.extend(results)
        offset += limit
        if not results or offset >= min(payload.get("total_count", 0), 5000):
            break
    return out


def fetch_traffic_state() -> list[dict]:
    """
    Return a list of segments: {state, label, color, coords:[[lat,lon],...]}.
    Raises LiveDataError if every mirror fails.
    """
    last_error: Exception | None = None
    for domain in _DOMAINS:
        try:
            records = _records(domain, _DATASET)
            segments = [_normalize(r) for r in records]
            segments = [s for s in segments if s and s["coords"]]
            if segments:
                return segments
        except Exception as exc:  # noqa: BLE001 - any failure -> try next mirror
            last_error = exc
    raise LiveDataError(f"No live mirror reachable ({last_error})")


def _normalize(rec: dict) -> dict | None:
    state_raw = rec.get("estat") if rec.get("estat") is not None else rec.get("estado")
    try:
        state = int(state_raw)
    except (TypeError, ValueError):
        state = 4
    label, color = TRAFFIC_STATE.get(state, TRAFFIC_STATE[4])

    geom = rec.get("geo_shape") or {}
    geom = geom.get("geometry", geom)
    gtype, coords = geom.get("type"), geom.get("coordinates")
    if not coords:
        return None

    lines: list[list] = []
    if gtype == "LineString":
        lines = [coords]
    elif gtype == "MultiLineString":
        lines = coords
    else:
        return None

    # Opendatasoft returns [lon, lat]; folium wants [lat, lon].
    latlon = [[[pt[1], pt[0]] for pt in line] for line in lines]
    flat = [pt for line in latlon for pt in line]
    return {
        "state": state,
        "label": label,
        "color": color,
        "name": rec.get("denominacion") or rec.get("denominacio") or "",
        "coords": flat,
        "lines": latlon,
    }
