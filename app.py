"""Geospatial Business Expansion & Location Intelligence Platform.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import mapping, raster_utils, reporting
from src import spatial_analysis as sa
from src import opportunity_scoring as osc
from src.pipeline import Filters, Params, analyze, prepare_data
from src.validation import (
    OUTSIDE_REGION,
    SALES_OPTIONAL,
    SALES_REQUIRED,
    STORES_OPTIONAL,
    STORES_REQUIRED,
    MissingDependencyError,
    ValidationError,
)

st.set_page_config(page_title="Location Intelligence", page_icon="📍", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    [data-testid="stMetricValue"] {font-size: 1.6rem;}
    @media (max-width: 640px) {.block-container {padding-left: 0.8rem; padding-right: 0.8rem;} [data-testid="stMetricValue"] {font-size: 1.25rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = ["Overview", "Map", "Opportunities", "Spatial analysis", "Data & upload", "Raster (COG)", "About"]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def sig(f) -> tuple | None:
    return None if f is None else (f.name, f.size)


def memo(key: str, signature, fn):
    """Remember the last result of ``fn`` for ``key`` while ``signature`` is unchanged."""
    store = st.session_state.setdefault("_memo", {})
    if store.get(key, (None, None))[0] != signature:
        store[key] = (signature, fn())
    return store[key][1]


def _stretch(fn, *args, **kwargs):
    """Call a Streamlit element stretched to the container width on old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def table(df, **kwargs):
    return _stretch(st.dataframe, df, hide_index=True, **kwargs)


def show_chart(fig, empty_msg: str):
    if fig is None:
        st.info(empty_msg)
    else:
        _stretch(st.plotly_chart, fig)


def show_map(m, height: int = 640):
    try:
        from streamlit_folium import st_folium
    except ImportError:
        st.warning("streamlit-folium is not installed; showing a static embed. Install with `pip install streamlit-folium`.")
        st.components.v1.html(m._repr_html_(), height=height)
        return
    try:
        st_folium(m, height=height, use_container_width=True, returned_objects=[])
    except TypeError:  # newer streamlit-folium versions dropped use_container_width
        st_folium(m, height=height, width="stretch", returned_objects=[])


def money(v: float) -> str:
    return f"{v:,.0f}"


# ----------------------------------------------------------------------------
# Sidebar: navigation, data source
# ----------------------------------------------------------------------------
st.sidebar.title("📍 Location Intelligence")
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()

st.sidebar.subheader("1 · Data")
mode = st.sidebar.radio("Source", ["Synthetic sample data", "Upload my own data"], label_visibility="collapsed")

auto_cell_km = 5.0
is_synthetic = mode.startswith("Synthetic")
input_sig = None

def load_inputs():
    """Read the selected inputs (parsing only; validation happens in prepare_data)."""
    if is_synthetic:
        return dl.load_sample_sales(), dl.load_sample_stores(), (None if use_grid else dl.load_sample_regions())
    if sales_file is None:
        return None, None, None
    sales = dl.load_points_table(sales_file)
    stores = dl.load_points_table(stores_file) if stores_file is not None else None
    regions = dl.load_regions(regions_file).data if regions_file is not None else None
    return sales, stores, regions


use_grid, sales_file, stores_file, regions_file = False, None, None, None
if is_synthetic:
    use_grid = st.sidebar.checkbox("Use auto grid instead of sample zones", value=False)
    if use_grid:
        auto_cell_km = st.sidebar.slider("Grid cell size (km)", 2.0, 20.0, 5.0, 0.5)
    input_sig = ("sample", use_grid, auto_cell_km)
else:
    kinds = ["csv", "xlsx", "geojson", "json", "parquet", "zip"]
    sales_file = st.sidebar.file_uploader("Sales / customers (required)", type=kinds)
    stores_file = st.sidebar.file_uploader("Stores (optional)", type=kinds)
    regions_file = st.sidebar.file_uploader("Region boundaries (optional)", type=["geojson", "json", "zip", "parquet"])
    if regions_file is None:
        auto_cell_km = st.sidebar.slider("No boundaries: auto grid cell (km)", 2.0, 20.0, 5.0, 0.5)
    input_sig = ("upload", sig(sales_file), sig(stores_file), sig(regions_file), auto_cell_km)

try:
    sales_df, stores_df, regions_gdf = memo("load", input_sig, load_inputs)
except (ValidationError, MissingDependencyError) as exc:
    st.sidebar.error(str(exc))
    st.stop()

if sales_df is None:
    st.title("Geospatial Business Expansion Platform")
    st.info(
        "⬅️ Upload a sales/customer file in the sidebar to begin, or switch the source to **Synthetic sample data**.\n\n"
        f"Required columns: `{', '.join(SALES_REQUIRED)}`. Optional: `{', '.join(SALES_OPTIONAL)}`. "
        "See **Data & upload** for details once data is loaded."
    )
    st.stop()

try:
    with st.spinner("Validating data and running spatial joins…"):
        prep = memo("prep", input_sig, lambda: prepare_data(sales_df, stores_df, regions_gdf, auto_cell_km, is_synthetic))
except (ValidationError, MissingDependencyError) as exc:
    st.error(f"**Could not prepare the data.** {exc}")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar: filters and settings
# ----------------------------------------------------------------------------
orders0 = prep.orders
st.sidebar.divider()
st.sidebar.subheader("2 · Filters")

date_range = None
if "date" in orders0.columns:
    dmin, dmax = orders0["date"].min().date(), orders0["date"].max().date()
    picked = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(picked, (tuple, list)) and len(picked) == 2 and (picked[0], picked[1]) != (dmin, dmax):
        date_range = (picked[0], picked[1])

region_opts = sorted(orders0["region"].unique())
sel_regions = st.sidebar.multiselect("Regions", region_opts, default=region_opts)
cats = None
if "category" in orders0.columns:
    cat_opts = sorted(orders0["category"].unique())
    sel_cats = st.sidebar.multiselect("Product categories", cat_opts, default=cat_opts)
    cats = None if len(sel_cats) == len(cat_opts) else sel_cats
smin, smax = float(orders0["sales"].min()), float(orders0["sales"].max())
sales_range = None
if smax > smin:
    sr = st.sidebar.slider("Order value range", smin, smax, (smin, smax))
    sales_range = None if sr == (smin, smax) else sr

filters = Filters(
    date_range=date_range,
    regions=None if len(sel_regions) == len(region_opts) else sel_regions,
    categories=cats,
    sales_range=sales_range,
)

st.sidebar.divider()
st.sidebar.subheader("3 · Analysis settings")
coverage_km = st.sidebar.slider("Store coverage radius (km)", 0.5, 30.0, 5.0, 0.5)
with st.sidebar.expander("Score weights", expanded=False):
    st.caption("Relative importance; automatically re-normalised to 100%.")
    w_demand = st.slider("Demand", 0, 100, int(osc.DEFAULT_WEIGHTS["demand"] * 100))
    w_gap = st.slider("Store gap (stores per demand)", 0, 100, int(osc.DEFAULT_WEIGHTS["store_gap"] * 100))
    w_dist = st.slider("Distance to nearest store", 0, 100, int(osc.DEFAULT_WEIGHTS["distance"] * 100))
    w_comp = st.slider("Low competition", 0, 100, int(osc.DEFAULT_WEIGHTS["competition"] * 100))
with st.sidebar.expander("Hotspots (DBSCAN) & candidates", expanded=False):
    use_db = st.checkbox("Enable DBSCAN hotspots", value=True)
    eps_km = st.slider("Cluster radius eps (km)", 0.5, 10.0, 2.0, 0.5)
    min_samples = st.slider("Min customers per hotspot", 3, 100, 15)
    top_n = st.slider("Candidate sites to suggest", 1, 10, 5)

params = Params(
    coverage_km=coverage_km,
    weights={"demand": w_demand, "store_gap": w_gap, "distance": w_dist, "competition": w_comp},
    top_n_sites=top_n,
    dbscan=use_db,
    dbscan_eps_km=eps_km,
    dbscan_min_samples=min_samples,
)

try:
    with st.spinner("Scoring regions…"):
        result = memo("result", (input_sig, repr(filters), repr(params)), lambda: analyze(prep, filters, params))
except ValidationError as exc:
    st.warning(str(exc))
    st.stop()
except MissingDependencyError as exc:
    st.error(str(exc))
    st.stop()

S = result.summary
st.sidebar.divider()
st.sidebar.caption("Decision-support estimates only. Not guaranteed business outcomes.")

# ----------------------------------------------------------------------------
# Page header
# ----------------------------------------------------------------------------
st.title(page if page != "Overview" else "Geospatial Business Expansion Platform")
if S["is_synthetic"]:
    st.warning("**Synthetic demonstration data.** Customers, stores and zones are invented; results are not real-world findings.", icon="🧪")

# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
if page == "Overview":
    st.caption(f"Region boundaries: {S['regions_source']} · Demand basis: {S['demand_basis']}")
    c = st.columns(5)
    with c[0], st.container(border=True):
        st.metric("Total revenue", money(S["revenue"]))
    with c[1], st.container(border=True):
        st.metric("Orders", f"{S['orders']:,}")
    with c[2], st.container(border=True):
        st.metric("Customers", f"{S['customers']:,}")
    with c[3], st.container(border=True):
        st.metric("Existing stores", S["own_stores"], f"{S['competitors']} competitors", delta_color="off")
    with c[4], st.container(border=True):
        st.metric("Opportunity regions", S["opportunity_regions"], f"of {S['regions']} regions", delta_color="off",
                  help="Regions with relatively high demand AND relatively few stores for that demand (normalised ≥ 0.5 on both).")

    c2 = st.columns(3)
    c2[0].metric("Customers covered", f"{S['covered_customer_share']:.0%}" if S["covered_customer_share"] is not None else "n/a",
                 help=f"Within {coverage_km:g} km of an own store")
    c2[1].metric("Mean distance to nearest store", f"{S['mean_nearest_store_km']:.1f} km" if S["mean_nearest_store_km"] is not None else "n/a")
    c2[2].metric("DBSCAN hotspots", S["hotspots"] if use_db else "off")

    if prep.warnings or result.notes:
        with st.expander(f"Data notes ({len(prep.warnings) + len(result.notes)})"):
            for w in prep.warnings + result.notes:
                st.write("• " + w)

    a, b = st.columns(2)
    with a:
        show_chart(reporting.fig_sales_by_region_category(result.orders), "No sales to chart.")
    with b:
        show_chart(reporting.fig_demand_vs_coverage(result.scored), "Add store data to compare demand with coverage.")
    a, b = st.columns(2)
    with a:
        show_chart(reporting.fig_top_opportunities(result.scored), "No regions to rank.")
    with b:
        show_chart(reporting.fig_sales_trend(result.orders), "Add a `date` column to see the sales trend.")

    st.download_button("⬇️ Download analysis summary (Markdown)", reporting.analysis_summary_text(result).encode("utf-8"),
                       "analysis_summary.md", "text/markdown")

elif page == "Map":
    st.caption("Use the layer control (top-right of the map) to toggle layers. Candidate sites are estimates.")
    cols = st.columns([2, 1, 1, 1, 1])
    metric = cols[0].selectbox("Colour regions by", list(mapping.CHOROPLETH_METRICS), format_func=mapping.CHOROPLETH_METRICS.get)
    show_cust = cols[1].checkbox("Customer points", value=False)
    show_heat = cols[2].checkbox("Heatmap", value=True)
    show_cov = cols[3].checkbox("Coverage", value=True)
    show_hot = cols[4].checkbox("Hotspots", value=False)
    with st.spinner("Rendering map…"):
        fmap = mapping.build_map(result, {"customers": show_cust, "heatmap": show_heat, "coverage": show_cov, "hotspots": show_hot}, metric)
    show_map(fmap)

elif page == "Opportunities":
    st.warning(osc.DISCLAIMER, icon="⚠️")
    w_used = osc.normalize_weights(params.weights, S["components_used"].split(", "))
    st.markdown("**Score** = 100 × Σ (weight × normalised component). Weights in use: "
                + ", ".join(f"{osc.COMPONENT_LABELS[k]} **{v:.0%}**" for k, v in w_used.items()))
    if set(w_used) != set(osc.COMPONENTS):
        missing = [osc.COMPONENT_LABELS[k] for k in osc.COMPONENTS if k not in w_used]
        st.info("Components excluded because the needed data is not available: " + ", ".join(missing))

    rec = reporting.recommendations_table(result.scored)
    table(
        rec, height=420,
        column_config={
            "opportunity_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "high_demand_low_presence": st.column_config.CheckboxColumn("High demand / low presence"),
            "demand_total": st.column_config.NumberColumn("Demand", format="%.0f"),
            "sales": st.column_config.NumberColumn("Sales", format="%.0f"),
            "uncovered_share": st.column_config.NumberColumn("Customers uncovered", format="percent"),
            "covered_area_share": st.column_config.NumberColumn("Area covered", format="percent"),
            "explanation": st.column_config.TextColumn("Why this score", width="large"),
        },
    )
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Recommendations (CSV)", csv_bytes(rec), "expansion_recommendations.csv", "text/csv")
    d2.download_button("⬇️ Candidate sites (CSV)", csv_bytes(result.candidates), "candidate_sites.csv", "text/csv")

    st.subheader("Why the top regions rank where they do")
    for _, r in result.scored.head(5).iterrows():
        with st.expander(f"#{int(r['rank'])} · {r['region']} · score {r['opportunity_score']:.1f}"):
            st.write(r["explanation"])
            st.caption("Measured values: " + f"{int(r['customers'])} customers, sales {r['sales']:,.0f}, "
                       f"{int(r['own_stores'])} own store(s), {int(r['competitors'])} competitor(s).")

    st.subheader("Suggested candidate locations (estimates)")
    st.caption("Sales-weighted centroid of customers outside current coverage in each top region. No rent, zoning or road-network data is used.")
    table(result.candidates)
    show_chart(reporting.fig_top_opportunities(result.scored), "No regions to rank.")

elif page == "Spatial analysis":
    t1, t2, t3, t4, t5 = st.tabs(["Spatial join & containment", "Nearest neighbour", "Coverage & buffers", "Intersection", "DBSCAN hotspots"])
    with t1:
        st.markdown("Each order/store point is matched to the region polygon that **contains** it (`geopandas.sjoin`, predicate `within`).")
        counts = result.scored[["region", "customers", "orders", "own_stores", "competitors"]]
        table(counts)
        n_out = int((result.orders["region"] == OUTSIDE_REGION).sum())
        st.caption(f"{n_out:,} order row(s) fall outside every polygon.")
        st.markdown("**Sample of joined order points**")
        table(result.orders[["order_id" if "order_id" in result.orders.columns else "customer_id", "latitude", "longitude", "region"]].head(15))
    with t2:
        if S["mean_nearest_store_km"] is None:
            st.info("Nearest-neighbour analysis needs at least one own store.")
        else:
            st.markdown("Distances are measured in a projected metric CRS (UTM), **not** in degrees.")
            st.caption(f"Projected CRS: {prep.crs}")
            show_chart(reporting.fig_nearest_distance(result.customers, coverage_km), "No distances to show.")
            per_store = (result.customers.groupby("nearest_store_id")
                         .agg(customers=("customer_id", "size"), sales=("sales", "sum"), mean_dist_km=("nearest_store_dist_km", "mean"),
                              max_dist_km=("nearest_store_dist_km", "max")).round(2).reset_index().sort_values("customers", ascending=False))
            st.markdown("**Customers whose nearest own store is…**")
            table(per_store)
    with t3:
        if result.coverage is None:
            st.info("Coverage analysis needs at least one own store.")
        else:
            st.markdown(f"Each own store gets a **{coverage_km:g} km buffer** (computed in metres); the union is the covered area. "
                        "A customer is *covered* if their nearest own store is within the radius.")
            area = sa.coverage_area_by_region(result.regions, result.coverage, crs=prep.crs)
            area = area.merge(result.scored[["region", "uncovered_share"]], on="region", how="left").round(3)
            cov_share = S["covered_customer_share"]
            m1, m2 = st.columns(2)
            m1.metric("Customers covered", f"{cov_share:.0%}")
            m2.metric("Customers not covered", f"{result.customers['covered'].eq(False).sum():,}")
            st.markdown("**Underserved areas** - regions sorted by share of customers outside coverage")
            table(area.sort_values("uncovered_share", ascending=False),
                         column_config={"covered_area_share": st.column_config.NumberColumn("Area covered", format="percent"),
                                        "uncovered_share": st.column_config.NumberColumn("Customers uncovered", format="percent")})
            show_chart(reporting.fig_customer_distribution(result.customers, result.stores), "No customers.")
    with t4:
        if result.coverage is None:
            st.info("Intersection analysis (regions × coverage) needs at least one own store.")
        else:
            pieces = sa.intersection_analysis(result.regions[["region", "geometry"]], result.coverage[["geometry"]], crs=prep.crs)
            st.markdown("`geopandas.overlay(how='intersection')` of **region polygons** with the **coverage polygon** - the parts of each region that are served.")
            table(pd.DataFrame(pieces.drop(columns="geometry")).round(2))
    with t5:
        if not use_db:
            st.info("Enable DBSCAN in the sidebar (Hotspots & candidates).")
        elif result.hotspots.empty:
            st.info("No hotspots found with these settings. Try a larger radius or fewer minimum customers.")
        else:
            st.markdown("DBSCAN clusters customers that are packed closely together; points that belong to no cluster are noise.")
            table(result.hotspots.round(4))
            noise = int((result.customers["hotspot"] < 0).sum())
            st.caption(f"{noise:,} customers are not part of any hotspot.")

elif page == "Data & upload":
    st.subheader("Expected schema")
    a, b = st.columns(2)
    a.markdown("**Sales / customers**\n\n" + "\n".join(f"- `{c}` (required)" for c in SALES_REQUIRED)
               + "\n" + "\n".join(f"- `{c}`" for c in SALES_OPTIONAL))
    b.markdown("**Stores**\n\n" + "\n".join(f"- `{c}` (required)" for c in STORES_REQUIRED)
               + "\n" + "\n".join(f"- `{c}`" for c in STORES_OPTIONAL) + "\n\n`store_type` must be `own` or `competitor`.")
    st.caption("Latitude/longitude must be WGS84 decimal degrees. Column names are case-insensitive; common aliases (lat, lng, revenue…) are recognised. "
               "Point files (GeoJSON/Shapefile zip/GeoParquet) are converted to latitude/longitude automatically.")
    st.subheader("Validation report")
    if prep.warnings:
        for w in prep.warnings:
            st.write("• " + w)
    else:
        st.success("No validation issues.")
    st.subheader("Data preview")
    t1, t2, t3 = st.tabs(["Orders", "Stores", "Regions"])
    with t1:
        table(prep.orders.head(200))
    with t2:
        table(prep.stores)
    with t3:
        table(pd.DataFrame(prep.regions.drop(columns="geometry")))

    st.subheader("Export")
    what = st.selectbox("Dataset", ["Customers (filtered)", "Orders (filtered)", "Stores", "Region scores", "Regions (polygons)"])
    fmt = st.selectbox("Format", list(dl.EXPORT_FORMATS))
    frames = {"Customers (filtered)": result.customers, "Orders (filtered)": result.orders, "Stores": result.stores,
              "Region scores": result.scored, "Regions (polygons)": result.regions}
    try:
        if what == "Regions (polygons)" and fmt in ("CSV", "Excel (.xlsx)"):
            raise ValidationError("Choose GeoJSON, Shapefile or GeoParquet to export polygons.")
        if what == "Region scores" and fmt not in ("CSV", "Excel (.xlsx)"):
            raise ValidationError("Region scores have no coordinates; export them as CSV or Excel.")
        data, fname, mime = dl.export_table(frames[what], fmt, basename=what.split(" ")[0].lower())
        st.download_button(f"⬇️ Download {fname}", data, fname, mime)
    except (ValidationError, MissingDependencyError) as exc:
        st.info(str(exc))

elif page == "Raster (COG)":
    st.markdown(
        "Overlay a **raster** (e.g. population density, night lights, elevation) on the map and sample its values at candidate sites. "
        "Only a decimated/windowed part is read, so large Cloud Optimized GeoTIFFs stay responsive."
    )
    if not raster_utils.rasterio_available():
        st.warning("`rasterio` is not installed, so raster features are disabled. Install it with `pip install rasterio`. The rest of the app is unaffected.")
    else:
        src_choice = st.radio("Raster source", ["Upload GeoTIFF", "HTTP(S) URL"], horizontal=True)
        raster_src = None
        if src_choice == "Upload GeoTIFF":
            up = st.file_uploader("GeoTIFF / COG", type=["tif", "tiff"])
            raster_src = up.getvalue() if up is not None else None
        else:
            url = st.text_input("URL of a GeoTIFF / COG (server must allow HTTP range requests)")
            raster_src = url.strip() or None
        limit = st.checkbox("Read only the study-area window", value=True,
                            help="Reads just the blocks that overlap your data. This is where a true COG is much faster.")
        band = st.number_input("Band", min_value=1, value=1, step=1)
        if raster_src is None:
            st.info("Provide a raster to continue. No raster is required for any other feature.")
        else:
            minx, miny, maxx, maxy = result.regions.total_bounds
            try:
                with st.spinner("Reading raster preview…"):
                    prev = raster_utils.read_preview(raster_src, band=int(band), bbox_wgs84=(minx, miny, maxx, maxy) if limit else None)
                info = prev["info"]
                (st.success if info["looks_like_cog"] else st.warning)(info["verdict"])
                with st.expander("Raster metadata"):
                    st.json({k: v for k, v in info.items() if k != "verdict"})
                st.caption(f"Preview stats (decimated): {prev['stats']}")
                overlay = {"image": raster_utils.colorize(prev["array"]), "bounds": prev["bounds"], "name": "Raster preview"}
                fmap = mapping.build_map(result, {"choropleth": False, "heatmap": False, "coverage": False}, raster=overlay)
                show_map(fmap, 560)
                if len(result.candidates):
                    cand = result.candidates.copy()
                    cand["raster_value"] = raster_utils.sample_at_points(raster_src, cand["longitude"], cand["latitude"], int(band))
                    st.markdown("**Raster value at candidate sites** (NaN = outside raster or nodata)")
                    table(cand[["rank", "region", "latitude", "longitude", "raster_value"]])
            except (ValidationError, MissingDependencyError) as exc:
                st.error(str(exc))

else:  # About
    st.markdown(
        """
### What this app does
Analyses geographic sales data to find regions with **high demand but low store presence**, ranks them with a transparent score,
and suggests candidate branch locations. All numbers come from the data you supply; scores and candidate sites are **estimates**.

### Vector vs raster data
| | Vector | Raster |
|---|---|---|
| Represents | Discrete features: points (stores), lines (roads), polygons (regions) | A grid of cells, each with a value (elevation, population, imagery) |
| Typical formats | GeoJSON, Shapefile, GeoParquet | GeoTIFF, **Cloud Optimized GeoTIFF (COG)** |
| Good for | Joins, containment, nearest neighbour, buffers | Continuous surfaces, satellite imagery, density maps |
| In this app | Customers, stores, regions | Optional overlay + value sampling |

**COG** = a GeoTIFF with internal tiling and overviews so clients can read just the part they need (windowed or HTTP range reads).
**GeoParquet** = columnar, compressed vector storage that fits the modern data stack.
This app *checks* for COG structure heuristically; not every `.tif` is cloud optimized.

### Method
1. **Validate** coordinates, columns and geometries; invalid rows are dropped and reported.
2. **Spatial join** orders and stores to region polygons (point-in-polygon).
3. **Project to UTM** (metres) and compute nearest own store per customer (KD-tree), buffers, and coverage.
4. **Score** each region: demand, store gap (own stores per unit demand), mean distance to nearest store, and competition - min-max normalised and weighted.
5. **Suggest sites**: sales-weighted centroid of customers outside coverage in top regions.

### Limitations
Scores are relative to the regions in view, rely on straight-line distance (not travel time), ignore rent, zoning, cannibalisation and seasonality,
and are only as good as the input data. Use them to prioritise where to investigate, not as a forecast.
"""
    )
