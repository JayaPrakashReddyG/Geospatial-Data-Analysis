"""Interactive Folium map: choropleth, markers, heatmap, coverage, hotspots, raster overlay.

Everything uses free tile servers (no API keys). Popup text is HTML-escaped so
uploaded data cannot inject markup into the page.
"""
from __future__ import annotations

from html import escape

import pandas as pd

from .validation import MissingDependencyError

# Metrics that can colour the choropleth: key -> (label, higher_is_better_colour_scale)
CHOROPLETH_METRICS = {
    "opportunity_score": "Opportunity score (0-100)",
    "sales": "Total sales",
    "customers": "Customers",
    "own_stores": "Own stores",
    "competitors": "Competitor stores",
    "mean_dist_km": "Mean distance to nearest own store (km)",
    "uncovered_share": "Share of customers outside coverage",
}

_COLORS = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]


def _folium():
    try:
        import folium
        from folium import plugins
        from branca.colormap import LinearColormap
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("folium is required for maps. Install with: pip install folium") from exc
    return folium, plugins, LinearColormap


def _fmt(v, spec=",.0f", na="n/a"):
    return na if v is None or (isinstance(v, float) and pd.isna(v)) else format(v, spec)


def _popup(title: str, rows: dict[str, str]) -> str:
    body = "".join(f"<tr><td style='padding-right:8px;color:#555'>{escape(k)}</td><td><b>{escape(str(v))}</b></td></tr>" for k, v in rows.items())
    return f"<div style='font-size:12px'><div style='font-weight:700;margin-bottom:4px'>{escape(title)}</div><table>{body}</table></div>"


def build_map(
    result,
    layers: dict[str, bool] | None = None,
    metric: str = "opportunity_score",
    max_customers: int = 1500,
    raster: dict | None = None,
):
    """Build the Folium map for an :class:`~src.pipeline.AnalysisResult`.

    ``layers`` toggles: choropleth, heatmap, customers, own_stores, competitors,
    candidates, coverage, hotspots (all default on except customers/hotspots).
    ``raster`` = {"image": RGBA ndarray, "bounds": [[s, w], [n, e]], "name": str} adds an image overlay.
    """
    folium, plugins, LinearColormap = _folium()
    layers = {
        "choropleth": True, "heatmap": True, "customers": False, "own_stores": True,
        "competitors": True, "candidates": True, "coverage": True, "hotspots": False,
        **(layers or {}),
    }

    regions = result.regions.reset_index(drop=True)
    minx, miny, maxx, maxy = regions.total_bounds
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=10, tiles="CartoDB positron", control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    # ---- choropleth -------------------------------------------------------
    if layers["choropleth"]:
        cols = ["region", "opportunity_score", "rank", "sales", "customers", "own_stores", "competitors",
                "mean_dist_km", "uncovered_share", "explanation"]
        data = result.scored[[c for c in cols if c in result.scored.columns]]
        gdf = regions[["region", "geometry"]].merge(data, on="region", how="left")
        if metric not in gdf.columns:
            metric = "opportunity_score"
        vals = pd.to_numeric(gdf[metric], errors="coerce")
        vmin, vmax = (float(vals.min()), float(vals.max())) if vals.notna().any() else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1.0
        cmap = LinearColormap(_COLORS, vmin=vmin, vmax=vmax, caption=CHOROPLETH_METRICS.get(metric, metric))

        for c in gdf.columns:
            if c not in ("geometry", "region", "explanation"):
                gdf[c] = pd.to_numeric(gdf[c], errors="coerce").round(3)
        fields = [c for c in ["region", "rank", "opportunity_score", "sales", "customers", "own_stores", "competitors"] if c in gdf.columns]
        aliases = {"region": "Region", "rank": "Rank", "opportunity_score": "Score", "sales": "Sales",
                   "customers": "Customers", "own_stores": "Own stores", "competitors": "Competitors"}

        def style(feature, _metric=metric, _cmap=cmap):
            v = feature["properties"].get(_metric)
            return {
                "fillColor": _cmap(v) if v is not None else "#cccccc",
                "color": "#444444", "weight": 1, "fillOpacity": 0.55,
            }

        folium.GeoJson(
            gdf.drop(columns=["explanation"], errors="ignore").to_json(),
            name="Regions (choropleth)",
            style_function=style,
            highlight_function=lambda f: {"weight": 3, "color": "#000000"},
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=[aliases[f] for f in fields], sticky=True),
        ).add_to(m)
        cmap.add_to(m)

    # ---- coverage buffers ---------------------------------------------------
    if layers["coverage"] and result.coverage is not None:
        folium.GeoJson(
            result.coverage[["geometry"]].to_json(),
            name=f"Own-store coverage ({result.params.coverage_km:g} km)",
            style_function=lambda f: {"fillColor": "#2c7fb8", "color": "#2c7fb8", "weight": 1.5, "fillOpacity": 0.12},
            interactive=False,
        ).add_to(m)

    # ---- heatmap ------------------------------------------------------------
    cust = result.customers
    if layers["heatmap"] and len(cust):
        w = cust["sales"].clip(lower=0)
        top = w.quantile(0.98) or 1.0
        pts = [[la, lo, float(min(x / top, 1.0))] for la, lo, x in zip(cust["latitude"], cust["longitude"], w)]
        plugins.HeatMap(pts, name="Demand heatmap (sales-weighted)", radius=18, blur=22, min_opacity=0.25).add_to(m)

    # ---- customers ----------------------------------------------------------
    if layers["customers"] and len(cust):
        sample = cust.sample(min(max_customers, len(cust)), random_state=0)
        fg = folium.FeatureGroup(name=f"Customers (sample of {len(sample):,})")
        for r in sample.itertuples():
            folium.CircleMarker(
                [r.latitude, r.longitude], radius=3, color="#555555", weight=0.5, fill=True,
                fill_color="#2ca25f" if r.covered else "#de2d26", fill_opacity=0.7,
                popup=folium.Popup(_popup(f"Customer {r.customer_id}", {
                    "Region": r.region, "Sales": _fmt(r.sales), "Orders": _fmt(r.orders),
                    "Nearest own store (km)": _fmt(r.nearest_store_dist_km, ".2f"),
                    "Covered": "yes" if r.covered else "no"}), max_width=260),
            ).add_to(fg)
        fg.add_to(m)

    # ---- stores -------------------------------------------------------------
    stores = result.stores
    for key, stype, color, icon, label in (
        ("own_stores", "own", "blue", "shopping-cart", "Own stores"),
        ("competitors", "competitor", "red", "flag", "Competitors"),
    ):
        if layers[key] and len(stores):
            sub = stores[stores["store_type"] == stype]
            if len(sub):
                fg = folium.FeatureGroup(name=label)
                for r in sub.itertuples():
                    folium.Marker(
                        [r.latitude, r.longitude], icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                        tooltip=escape(str(r.store_name)),
                        popup=folium.Popup(_popup(str(r.store_name), {"ID": r.store_id, "Type": stype, "Region": r.region}), max_width=240),
                    ).add_to(fg)
                fg.add_to(m)

    # ---- candidate sites ----------------------------------------------------
    if layers["candidates"] and len(result.candidates):
        fg = folium.FeatureGroup(name="Candidate sites (estimates)")
        for r in result.candidates.itertuples():
            folium.Marker(
                [r.latitude, r.longitude], icon=folium.Icon(color="green", icon="star", prefix="fa"),
                tooltip=f"Candidate #{r.rank}: {escape(str(r.region))}",
                popup=folium.Popup(_popup(f"Candidate #{r.rank} - {r.region}", {
                    "Opportunity score": _fmt(r.opportunity_score, ".1f"),
                    "Customers considered": _fmt(r.customers_considered),
                    "Basis": r.location_basis}), max_width=300),
            ).add_to(fg)
        fg.add_to(m)

    # ---- DBSCAN hotspots ----------------------------------------------------
    if layers["hotspots"] and len(result.hotspots):
        fg = folium.FeatureGroup(name="DBSCAN demand hotspots")
        vmax = result.hotspots["value"].max() or 1.0
        for r in result.hotspots.itertuples():
            folium.CircleMarker(
                [r.latitude, r.longitude], radius=8 + 14 * (r.value / vmax), color="#6a3d9a", weight=2,
                fill=True, fill_color="#6a3d9a", fill_opacity=0.25,
                popup=folium.Popup(_popup(f"Hotspot {r.hotspot}", {"Customers": _fmt(r.n_customers), "Sales": _fmt(r.value)}), max_width=220),
            ).add_to(fg)
        fg.add_to(m)

    # ---- raster overlay -----------------------------------------------------
    if raster is not None:
        folium.raster_layers.ImageOverlay(
            image=raster["image"], bounds=raster["bounds"], name=raster.get("name", "Raster"), opacity=1.0, zindex=2
        ).add_to(m)

    m.get_root().html.add_child(folium.Element(_legend_html(layers)))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def _legend_html(layers: dict[str, bool]) -> str:
    items = [("#1f78b4", "Own store"), ("#e31a1c", "Competitor"), ("#33a02c", "Candidate site (estimate)")]
    if layers.get("customers"):
        items += [("#2ca25f", "Customer - covered"), ("#de2d26", "Customer - not covered")]
    if layers.get("hotspots"):
        items.append(("#6a3d9a", "DBSCAN hotspot"))
    rows = "".join(
        f"<div><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};margin-right:6px'></span>{escape(t)}</div>"
        for c, t in items
    )
    return (
        "<div style='position:fixed;bottom:28px;left:12px;z-index:9999;background:rgba(255,255,255,.92);"
        "padding:8px 10px;border:1px solid #bbb;border-radius:6px;font:12px/1.5 sans-serif;color:#222'>"
        "<b>Legend</b>" + rows + "</div>"
    )
