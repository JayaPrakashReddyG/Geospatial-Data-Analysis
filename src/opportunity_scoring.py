"""Transparent opportunity scoring for candidate expansion regions.

Score = 100 x sum_i( w_i x c_i ), where each component c_i is min-max
normalised to [0, 1] ACROSS THE REGIONS BEING COMPARED (so scores are relative
rankings, not absolute probabilities):

========== ==================================================== =================
component  raw measure                                          higher c_i means
========== ==================================================== =================
demand     sum of ``demand`` (or ``sales`` as a proxy)          more demand
store_gap  1 - norm(own stores per unit of demand)              fewer stores for the demand
distance   mean customer distance to nearest own store (km)     customers are farther away
competition 1 - norm(number of competitor stores)               fewer competitors
========== ==================================================== =================

Components whose inputs do not exist (no own stores, no competitor rows) are
EXCLUDED and the remaining weights are re-normalised - nothing is invented.

This module only needs pandas/numpy so it is easy to unit-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import OUTSIDE_REGION

COMPONENTS = ("demand", "store_gap", "distance", "competition")
DEFAULT_WEIGHTS = {"demand": 0.35, "store_gap": 0.25, "distance": 0.25, "competition": 0.15}
COMPONENT_LABELS = {
    "demand": "High demand",
    "store_gap": "Few own stores for the demand",
    "distance": "Customers far from existing stores",
    "competition": "Low competition",
}
DISCLAIMER = (
    "Decision-support estimate only. Scores rank regions relative to each other using the supplied data; "
    "they are not forecasts and do not guarantee any business outcome. Validate with local market research."
)


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------
def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max scale to [0, 1]. A constant series (no information) maps to a neutral 0.5."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - lo) / (hi - lo)
    if invert:
        out = 1.0 - out
    return out.where(s.notna(), np.nan)


def normalize_weights(weights: dict[str, float], available: tuple[str, ...] | list[str] | None = None) -> dict[str, float]:
    """Restrict to ``available`` components and rescale to sum to 1.

    Raises ValueError for negative weights. If every available weight is zero,
    equal weights are used.
    """
    available = list(available) if available is not None else list(COMPONENTS)
    if any(float(v) < 0 for v in weights.values()):
        raise ValueError("Weights must be non-negative.")
    w = {k: float(weights.get(k, 0.0)) for k in available}
    total = sum(w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w} if w else {}
    return {k: v / total for k, v in w.items()}


# --------------------------------------------------------------------------
# Feature table per region
# --------------------------------------------------------------------------
def build_region_features(
    customers: pd.DataFrame,
    stores: pd.DataFrame,
    region_ids: list[str],
    area_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate customer- and store-level data to one row per region.

    ``customers`` needs: region, sales, orders; optional demand, nearest_store_dist_km, covered.
    ``stores`` needs: region, store_type ('own'/'competitor'); may be empty.
    ``area_table`` (optional) has region, area_km2, covered_area_share.
    """
    base = pd.DataFrame({"region": [r for r in dict.fromkeys(map(str, region_ids)) if r != OUTSIDE_REGION]})

    cust = customers[customers["region"].isin(base["region"])]
    use_demand = "demand" in cust.columns and cust["demand"].notna().any()
    agg = {"customers": ("customer_id", "nunique"), "orders": ("orders", "sum"), "sales": ("sales", "sum")}
    if use_demand:
        agg["demand_total"] = ("demand", "sum")
    if "nearest_store_dist_km" in cust.columns:
        agg["mean_dist_km"] = ("nearest_store_dist_km", "mean")
    if "covered" in cust.columns:
        agg["covered_customers"] = ("covered", "sum")
    grouped = cust.groupby("region").agg(**agg).reset_index() if len(cust) else pd.DataFrame(columns=["region"])
    feat = base.merge(grouped, on="region", how="left")

    for col in ("customers", "orders", "sales"):
        feat[col] = feat.get(col, 0)
        feat[col] = feat[col].fillna(0)
    feat["demand_total"] = feat["demand_total"].fillna(0) if use_demand else feat["sales"]
    feat["demand_basis"] = "demand column" if use_demand else "sales (proxy for demand)"

    if "covered_customers" in feat.columns:
        feat["uncovered_share"] = np.where(
            feat["customers"] > 0, 1.0 - feat["covered_customers"] / feat["customers"].replace(0, np.nan), np.nan
        )
    else:
        feat["uncovered_share"] = np.nan
    if "mean_dist_km" not in feat.columns:
        feat["mean_dist_km"] = np.nan

    if len(stores) and "region" in stores.columns:
        counts = (
            stores.groupby(["region", "store_type"]).size().unstack(fill_value=0)
            if "store_type" in stores.columns
            else pd.DataFrame()
        )
        counts = counts.reindex(columns=["own", "competitor"], fill_value=0)
        counts.columns = ["own_stores", "competitors"]
        feat = feat.merge(counts, left_on="region", right_index=True, how="left")
    feat["own_stores"] = feat.get("own_stores", 0)
    feat["competitors"] = feat.get("competitors", 0)
    feat[["own_stores", "competitors"]] = feat[["own_stores", "competitors"]].fillna(0).astype(int)

    if area_table is not None and len(area_table):
        feat = feat.merge(area_table[["region", "area_km2", "covered_area_share"]], on="region", how="left")
    return feat


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_regions(
    features: pd.DataFrame,
    weights: dict[str, float] | None = None,
    has_own_stores: bool = True,
    has_competitors: bool = True,
) -> pd.DataFrame:
    """Score and rank regions. Returns a copy of ``features`` with score columns added.

    Added columns: ``c_<component>`` (0-1), ``pts_<component>`` (points contributed),
    ``opportunity_score`` (0-100), ``rank`` (1 = best), ``explanation``,
    ``high_demand_low_presence`` (bool), and ``components_used``.
    """
    if features is None or len(features) == 0:
        raise ValueError("No regions to score.")
    df = features.copy().reset_index(drop=True)

    available = ["demand"]
    if has_own_stores:
        available += ["store_gap", "distance"]
    if has_competitors:
        available += ["competition"]
    w = normalize_weights(weights or DEFAULT_WEIGHTS, available)

    df["c_demand"] = minmax(df["demand_total"])
    if has_own_stores:
        density = df["own_stores"] / df["demand_total"].where(df["demand_total"] > 0)
        df["c_store_gap"] = (1.0 - minmax(density)).where(density.notna(), 0.0)
        df["c_distance"] = minmax(df["mean_dist_km"]).fillna(0.0)
    else:
        df["c_store_gap"] = np.nan
        df["c_distance"] = np.nan
    df["c_competition"] = minmax(df["competitors"], invert=True) if has_competitors else np.nan

    score = pd.Series(0.0, index=df.index)
    for comp in COMPONENTS:
        pts = 100.0 * w.get(comp, 0.0) * df[f"c_{comp}"].fillna(0.0)
        df[f"pts_{comp}"] = pts if comp in w else 0.0
        score += df[f"pts_{comp}"]
    df["opportunity_score"] = score.round(2)
    df["rank"] = df["opportunity_score"].rank(ascending=False, method="first").astype(int)

    df["components_used"] = ", ".join(w)
    df["explanation"] = [_explain(row, w) for _, row in df.iterrows()]
    df = df.sort_values("rank").reset_index(drop=True)
    return flag_opportunity_regions(df)


def flag_opportunity_regions(scored: pd.DataFrame, demand_thr: float = 0.5, gap_thr: float = 0.5) -> pd.DataFrame:
    """Mark regions with relatively high demand AND relatively low store presence.

    Thresholds apply to the normalised components (0-1), so the flag is relative
    to the other regions in the current view.
    """
    out = scored.copy()
    gap = out["c_store_gap"] if out["c_store_gap"].notna().any() else out["c_distance"]
    out["high_demand_low_presence"] = (out["c_demand"] >= demand_thr) & (gap.fillna(0) >= gap_thr)
    return out


def _detail(comp: str, row: pd.Series) -> str:
    if comp == "demand":
        return f"{row['demand_total']:,.0f} {'units' if row['demand_basis'] == 'demand column' else 'sales'}"
    if comp == "store_gap":
        return f"{int(row['own_stores'])} own store(s)"
    if comp == "distance":
        d = row.get("mean_dist_km")
        u = row.get("uncovered_share")
        txt = f"avg {d:.1f} km to nearest store" if pd.notna(d) else "no customers"
        if pd.notna(u):
            txt += f", {u:.0%} of customers outside coverage"
        return txt
    return f"{int(row['competitors'])} competitor store(s)"


def _explain(row: pd.Series, weights: dict[str, float]) -> str:
    """Plain-language reason for a region's score, built only from its measured values.

    "Strongest drivers" = components contributing the most points (weight x value).
    "Held back by" = the component with the lowest normalised value, shown only
    when it is below 0.5 (i.e. below the middle of the regions being compared).
    """
    contrib = sorted(((row[f"pts_{c}"], c) for c in weights), reverse=True)
    top = [f"{COMPONENT_LABELS[c]} ({p:.1f} pts; {_detail(c, row)})" for p, c in contrib[:2] if p > 0]
    text = f"Score {row['opportunity_score']:.1f}/100."
    if top:
        text += " Strongest drivers: " + "; ".join(top) + "."
    lowest = min(weights, key=lambda c: row[f"c_{c}"] if pd.notna(row[f"c_{c}"]) else 0.0)
    low_val = row[f"c_{lowest}"] if pd.notna(row[f"c_{lowest}"]) else 0.0
    if len(weights) > 1 and low_val < 0.5:
        text += f" Held back by: {COMPONENT_LABELS[lowest]} (normalised {low_val:.2f}; {_detail(lowest, row)})."
    return text


# --------------------------------------------------------------------------
# Candidate sites
# --------------------------------------------------------------------------
def suggest_candidate_sites(
    customers: pd.DataFrame, scored: pd.DataFrame, top_n: int = 5, weight_col: str = "sales"
) -> pd.DataFrame:
    """Propose one candidate location per top-ranked region.

    The location is the value-weighted centroid of customers who are OUTSIDE
    current coverage (or of all customers in the region if everyone is covered).
    This is a heuristic ESTIMATE of where a new branch would be closest to
    unserved demand - not a site-selection guarantee (no rent, zoning or
    road-network data is used). Weighted lat/lon averaging is fine at city scale.
    """
    rows = []
    for _, r in scored.sort_values("rank").iterrows():
        if len(rows) >= top_n:
            break
        g = customers[customers["region"] == r["region"]]
        if g.empty:
            continue
        basis = "all customers in region"
        if "covered" in g.columns and (~g["covered"].astype(bool)).any():
            g = g[~g["covered"].astype(bool)]
            basis = "customers outside current coverage"
        w = g[weight_col].to_numpy(float) if weight_col in g.columns else np.ones(len(g))
        if w.sum() <= 0:
            w = np.ones(len(g))
        rows.append(
            {
                "rank": int(r["rank"]),
                "region": r["region"],
                "latitude": float(np.average(g["latitude"], weights=w)),
                "longitude": float(np.average(g["longitude"], weights=w)),
                "opportunity_score": float(r["opportunity_score"]),
                "customers_considered": int(len(g)),
                "location_basis": f"sales-weighted centroid of {basis} (estimate)",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "rank", "region", "latitude", "longitude", "opportunity_score",
            "customers_considered", "location_basis",
        ],
    )
