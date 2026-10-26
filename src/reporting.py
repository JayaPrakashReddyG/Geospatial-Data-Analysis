"""Charts (Plotly), KPI helpers and downloadable summary text.

Chart functions return ``None`` when the data needed for them is missing, so
the dashboard can show an explanatory empty state instead of an error.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .opportunity_scoring import COMPONENT_LABELS, DISCLAIMER

PALETTE = ["#1f6feb", "#f78166", "#3fb950", "#d29922", "#a371f7", "#db61a2", "#79c0ff"]
LAYOUT = dict(margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13), template="plotly_white", colorway=PALETTE)


def _px():
    import plotly.express as px

    return px


def _style(fig, title: str, height: int = 380):
    fig.update_layout(title=dict(text=title, x=0.01), height=height, **LAYOUT)
    return fig


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def fig_sales_by_region_category(orders: pd.DataFrame):
    df = orders[orders["region"] != "Outside regions"]
    if df.empty:
        return None
    px = _px()
    if "category" in df.columns:
        agg = df.groupby(["region", "category"], as_index=False)["sales"].sum()
        order = agg.groupby("region")["sales"].sum().sort_values(ascending=False).index
        fig = px.bar(agg, x="region", y="sales", color="category", category_orders={"region": list(order)},
                     labels={"sales": "Sales", "region": "Region", "category": "Category"})
        return _style(fig, "Sales by region and category")
    agg = df.groupby("region", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    return _style(px.bar(agg, x="region", y="sales", labels={"sales": "Sales", "region": "Region"}), "Sales by region")


def fig_demand_vs_coverage(scored: pd.DataFrame):
    if scored.empty or scored["own_stores"].isna().all():
        return None
    px = _px()
    df = scored.copy()
    df["Opportunity region"] = df["high_demand_low_presence"].map({True: "High demand, low presence", False: "Other"})
    fig = px.scatter(
        df, x="demand_total", y="own_stores", size="customers", color="Opportunity region", hover_name="region",
        color_discrete_map={"High demand, low presence": "#e31a1c", "Other": "#7f8c8d"},
        labels={"demand_total": f"Demand ({df['demand_basis'].iloc[0]})", "own_stores": "Own stores in region"},
    )
    fig.update_yaxes(dtick=1, rangemode="tozero")
    return _style(fig, "Demand versus store coverage")


def fig_top_opportunities(scored: pd.DataFrame, top_n: int = 10):
    if scored.empty:
        return None
    px = _px()
    top = scored.nsmallest(top_n, "rank")
    comps = [c for c in ("demand", "store_gap", "distance", "competition") if (top[f"pts_{c}"] > 0).any()]
    if not comps:
        return None
    long = top.melt(id_vars=["region", "rank"], value_vars=[f"pts_{c}" for c in comps], var_name="component", value_name="points")
    long["component"] = long["component"].str.replace("pts_", "", regex=False).map(COMPONENT_LABELS)
    fig = px.bar(long, y="region", x="points", color="component", orientation="h",
                 category_orders={"region": top["region"].tolist()[::-1]},
                 labels={"points": "Score points", "region": "Region", "component": "Component"})
    fig.update_yaxes(autorange="reversed")
    return _style(fig, f"Top {len(top)} expansion opportunities (score breakdown)", height=max(320, 36 * len(top) + 120))


def fig_customer_distribution(customers: pd.DataFrame, stores: pd.DataFrame):
    if customers.empty:
        return None
    px = _px()
    fig = px.scatter(customers, x="longitude", y="latitude", color=customers["covered"].map({True: "Covered", False: "Not covered"}),
                     color_discrete_map={"Covered": "#3fb950", "Not covered": "#e31a1c"}, opacity=0.55,
                     labels={"color": "Coverage"}, hover_data={"customer_id": True, "sales": ":,.0f"})
    fig.update_traces(marker=dict(size=4))
    if len(stores):
        for stype, color, sym in (("own", "#1f6feb", "diamond"), ("competitor", "#000000", "x")):
            sub = stores[stores["store_type"] == stype]
            if len(sub):
                fig.add_scatter(x=sub["longitude"], y=sub["latitude"], mode="markers", name=f"{stype} stores",
                                marker=dict(size=11, color=color, symbol=sym, line=dict(width=1, color="white")),
                                text=sub["store_name"], hoverinfo="text")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return _style(fig, "Geographic customer distribution (lon/lat, equal scale)", height=460)


def fig_nearest_distance(customers: pd.DataFrame, coverage_km: float):
    d = customers["nearest_store_dist_km"].dropna()
    if d.empty:
        return None
    px = _px()
    fig = px.histogram(customers.dropna(subset=["nearest_store_dist_km"]), x="nearest_store_dist_km", nbins=40,
                       labels={"nearest_store_dist_km": "Distance to nearest own store (km)"})
    fig.add_vline(x=coverage_km, line_dash="dash", line_color="#e31a1c", annotation_text=f"coverage radius {coverage_km:g} km")
    fig.update_yaxes(title="Customers")
    return _style(fig, "Nearest-store distance distribution")


def fig_sales_trend(orders: pd.DataFrame):
    if "date" not in orders.columns or orders.empty:
        return None
    px = _px()
    m = orders.assign(month=orders["date"].dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False)["sales"].sum()
    return _style(px.line(m, x="month", y="sales", markers=True, labels={"month": "Month", "sales": "Sales"}), "Monthly sales trend")


# --------------------------------------------------------------------------
# Tables and text
# --------------------------------------------------------------------------
RECOMMENDATION_COLUMNS = [
    "rank", "region", "opportunity_score", "high_demand_low_presence", "demand_total", "demand_basis", "customers",
    "sales", "own_stores", "competitors", "mean_dist_km", "uncovered_share", "covered_area_share", "explanation",
]


def recommendations_table(scored: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in RECOMMENDATION_COLUMNS if c in scored.columns]
    out = scored[cols].copy()
    for c in ("mean_dist_km", "uncovered_share", "covered_area_share"):
        if c in out.columns:
            out[c] = out[c].round(3)
    out["note"] = "Decision-support estimate; not a guaranteed outcome."
    return out


def analysis_summary_text(result, weights_used: dict[str, float] | None = None) -> str:
    """Concise Markdown summary suitable for download."""
    s = result.summary
    top = result.scored.head(5)
    lines = [
        "# Geospatial Business Expansion - Analysis Summary",
        f"_Generated {date.today().isoformat()}_",
        "",
    ]
    if s["is_synthetic"]:
        lines += ["> **SYNTHETIC DEMONSTRATION DATA** - invented for demo purposes; findings are not real-world facts.", ""]
    lines += [
        "## Headline figures (from the data supplied)",
        f"- Revenue: {s['revenue']:,.0f}  |  Orders: {s['orders']:,}  |  Customers: {s['customers']:,}",
        f"- Own stores: {s['own_stores']}  |  Competitor stores: {s['competitors']}  |  Regions analysed: {s['regions']}",
    ]
    if s["covered_customer_share"] is not None:
        lines.append(
            f"- Customers within {result.params.coverage_km:g} km of an own store: {s['covered_customer_share']:.0%} "
            f"(mean distance to nearest own store: {s['mean_nearest_store_km']:.1f} km)"
        )
    lines += [
        f"- High-demand / low-presence regions: {s['opportunity_regions']}",
        f"- Region boundaries: {s['regions_source']}",
        "",
        "## Top ranked regions (scores are ESTIMATES)",
    ]
    for _, r in top.iterrows():
        lines.append(f"{int(r['rank'])}. **{r['region']}** - {r['explanation']}")
    lines += [
        "",
        "## Method",
        f"- Demand basis: {s['demand_basis']}.",
        f"- Score components used: {s['components_used']}. Weights are re-normalised over the components available.",
        f"- Weights (as configured): {weights_used or result.params.weights}",
        f"- Distances/buffers computed in a projected metric CRS; coverage radius {result.params.coverage_km:g} km.",
        "- Candidate sites are sales-weighted centroids of uncovered customers - heuristics, not site surveys.",
        "",
        f"**{DISCLAIMER}**",
    ]
    return "\n".join(lines)
