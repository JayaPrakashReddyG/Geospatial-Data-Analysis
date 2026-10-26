"""End-to-end orchestration: validate -> spatial join -> nearest store -> filter -> score.

Two stages keep the dashboard fast:

``prepare_data``  (expensive, depends only on the uploaded data)
    validation, region assignment (spatial join), nearest-store distances.
``analyze``       (cheaper, depends on sidebar filters and settings)
    filtering, customer table, coverage, region scoring, hotspots, candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import opportunity_scoring as osc
from . import spatial_analysis as sa
from .validation import (
    OUTSIDE_REGION,
    MissingDependencyError,
    ValidationError,
    validate_sales_df,
    validate_stores_df,
)


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class PreparedData:
    orders: pd.DataFrame            # validated order rows + region + nearest-store distance
    stores: pd.DataFrame            # validated stores + region (may be empty)
    regions: object                 # GeoDataFrame, EPSG:4326, columns region/geometry
    regions_source: str
    crs: str                        # projected CRS used for metric calculations
    warnings: list[str] = field(default_factory=list)
    is_synthetic: bool = False


@dataclass
class Filters:
    date_range: tuple | None = None          # (start, end) Timestamps or dates, inclusive
    regions: list[str] | None = None
    categories: list[str] | None = None
    sales_range: tuple[float, float] | None = None


@dataclass
class Params:
    coverage_km: float = 5.0
    weights: dict = field(default_factory=lambda: dict(osc.DEFAULT_WEIGHTS))
    top_n_sites: int = 5
    dbscan: bool = True
    dbscan_eps_km: float = 2.0
    dbscan_min_samples: int = 15


@dataclass
class AnalysisResult:
    orders: pd.DataFrame
    customers: pd.DataFrame
    stores: pd.DataFrame
    regions: object
    features: pd.DataFrame
    scored: pd.DataFrame
    candidates: pd.DataFrame
    coverage: object | None
    hotspots: pd.DataFrame
    summary: dict
    params: Params
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------
def prepare_data(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame | None = None,
    regions_gdf=None,
    auto_cell_km: float = 5.0,
    is_synthetic: bool = False,
) -> PreparedData:
    """Validate inputs and run the spatial joins that do not depend on filters."""
    notes: list[str] = []

    sres = validate_sales_df(sales_df)
    notes += sres.warnings
    orders = sres.data

    stores = None
    if stores_df is not None and len(stores_df) > 0:
        stres = validate_stores_df(stores_df)
        notes += stres.warnings
        stores = stres.data
    else:
        notes.append("No store data supplied: store coverage, distance and competition cannot be analysed.")

    orders_gdf = sa.points_to_gdf(orders)
    stores_gdf = sa.points_to_gdf(stores) if stores is not None else None

    all_pts = orders_gdf[["geometry"]]
    if stores_gdf is not None:
        all_pts = pd.concat([all_pts, stores_gdf[["geometry"]]], ignore_index=True)
        all_pts = sa._gpd().GeoDataFrame(all_pts, geometry="geometry", crs=sa.WGS84)
    crs = sa.choose_projected_crs(all_pts)

    if regions_gdf is None:
        regions_gdf = sa.make_grid_regions(all_pts, cell_km=auto_cell_km)
        source = f"auto-generated {auto_cell_km:g} km grid (analysis cells, NOT administrative boundaries)"
    else:
        source = "supplied boundary file"

    orders_gdf = sa.assign_regions(orders_gdf, regions_gdf)
    n_out = int((orders_gdf["region"] == OUTSIDE_REGION).sum())
    if n_out:
        notes.append(
            f"{n_out} of {len(orders_gdf)} order rows fall outside every region polygon; they count towards "
            "totals but not towards any region's score."
        )
    if "region_input" in orders_gdf.columns:
        mism = int((orders_gdf["region_input"] != orders_gdf["region"]).sum())
        if mism:
            notes.append(
                f"{mism} order row(s) have a 'region' value that differs from the polygon they fall in; "
                "the polygon (spatial join) result is used."
            )

    moves = orders.groupby("customer_id")[["latitude", "longitude"]].nunique().gt(1).any(axis=1).sum()
    if moves:
        notes.append(
            f"{int(moves)} customer_id(s) appear at more than one location; the first location is used per customer."
        )

    if stores_gdf is not None:
        stores_gdf = sa.assign_regions(stores_gdf, regions_gdf)
        own = stores_gdf[stores_gdf["store_type"] == "own"]
    else:
        own = None

    if own is not None and len(own):
        orders_gdf = sa.add_nearest_store(orders_gdf, own, crs=crs)
    else:
        orders_gdf["nearest_store_id"] = None
        orders_gdf["nearest_store_dist_km"] = np.nan
        if stores_gdf is not None:
            notes.append("Store data contains no own stores: distance and coverage metrics are unavailable.")

    orders_out = pd.DataFrame(orders_gdf.drop(columns="geometry"))
    stores_out = pd.DataFrame(stores_gdf.drop(columns="geometry")) if stores_gdf is not None else _empty_stores()
    return PreparedData(orders_out, stores_out, regions_gdf, source, crs, notes, is_synthetic)


def _empty_stores() -> pd.DataFrame:
    from .validation import empty_stores_frame

    out = empty_stores_frame()
    out["region"] = out["region"].astype(str)
    return out


# --------------------------------------------------------------------------
# Filtering and customer table (pure pandas)
# --------------------------------------------------------------------------
def apply_filters(orders: pd.DataFrame, f: Filters) -> pd.DataFrame:
    """Apply sidebar filters. Filters whose column is absent are ignored."""
    mask = pd.Series(True, index=orders.index)
    if f.date_range and "date" in orders.columns:
        start, end = pd.Timestamp(f.date_range[0]), pd.Timestamp(f.date_range[1])
        mask &= (orders["date"] >= start) & (orders["date"] <= end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    if f.regions is not None:
        mask &= orders["region"].isin(f.regions)
    if f.categories is not None and "category" in orders.columns:
        mask &= orders["category"].isin(f.categories)
    if f.sales_range:
        mask &= orders["sales"].between(f.sales_range[0], f.sales_range[1])
    return orders.loc[mask].reset_index(drop=True)


def build_customers(orders: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: location, region, total sales, order count, nearest-store distance."""
    agg = {
        "latitude": ("latitude", "first"),
        "longitude": ("longitude", "first"),
        "region": ("region", "first"),
        "sales": ("sales", "sum"),
        "orders": ("sales", "size"),
        "nearest_store_dist_km": ("nearest_store_dist_km", "first"),
        "nearest_store_id": ("nearest_store_id", "first"),
    }
    if "demand" in orders.columns:
        agg["demand"] = ("demand", "sum")
    return orders.groupby("customer_id", sort=False).agg(**agg).reset_index()


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------
def analyze(prep: PreparedData, filters: Filters | None = None, params: Params | None = None) -> AnalysisResult:
    filters = filters or Filters()
    params = params or Params()
    notes: list[str] = []

    orders = apply_filters(prep.orders, filters)
    if orders.empty:
        raise ValidationError("No orders match the current filters. Widen the date, region, category or sales filter.")
    customers = build_customers(orders)

    stores_all = prep.stores
    own_all = stores_all[stores_all["store_type"] == "own"] if len(stores_all) else stores_all
    has_own = len(own_all) > 0
    has_comp = len(stores_all) > 0 and bool((stores_all["store_type"] == "competitor").any())

    # Coverage: distance-based flag + dissolved buffer polygon + area share per region
    coverage = None
    area_table = None
    if has_own:
        customers["covered"] = customers["nearest_store_dist_km"] <= params.coverage_km
        coverage = sa.coverage_union(sa.points_to_gdf(own_all), params.coverage_km, crs=prep.crs)
        area_table = sa.coverage_area_by_region(prep.regions, coverage, crs=prep.crs)
    else:
        customers["covered"] = False

    regions = prep.regions
    stores_view = stores_all
    if filters.regions is not None:
        regions = regions[regions["region"].isin(filters.regions)].reset_index(drop=True)
        stores_view = stores_all[stores_all["region"].isin(filters.regions)] if len(stores_all) else stores_all
    if regions.empty:
        raise ValidationError("The region filter excludes every region that has boundaries.")

    features = osc.build_region_features(customers, stores_view, regions["region"].tolist(), area_table)
    scored = osc.score_regions(features, params.weights, has_own_stores=has_own, has_competitors=has_comp)
    candidates = osc.suggest_candidate_sites(customers, scored, params.top_n_sites)

    hotspots = pd.DataFrame()
    if params.dbscan and len(customers) >= max(params.dbscan_min_samples, 2):
        try:
            labels, hotspots = sa.dbscan_hotspots(
                customers, params.dbscan_eps_km, params.dbscan_min_samples, value_col="sales"
            )
            customers["hotspot"] = labels
        except MissingDependencyError as exc:
            notes.append(str(exc))
    if "hotspot" not in customers.columns:
        customers["hotspot"] = -1

    n_covered = int(customers["covered"].sum()) if has_own else None
    summary = {
        "revenue": float(orders["sales"].sum()),
        "orders": int(len(orders)),
        "customers": int(len(customers)),
        "own_stores": int((stores_view["store_type"] == "own").sum()) if len(stores_view) else 0,
        "competitors": int((stores_view["store_type"] == "competitor").sum()) if len(stores_view) else 0,
        "regions": int(len(scored)),
        "opportunity_regions": int(scored["high_demand_low_presence"].sum()),
        "covered_customer_share": (n_covered / len(customers)) if has_own else None,
        "mean_nearest_store_km": float(customers["nearest_store_dist_km"].mean()) if has_own else None,
        "demand_basis": scored["demand_basis"].iloc[0],
        "components_used": scored["components_used"].iloc[0],
        "hotspots": int(len(hotspots)),
        "is_synthetic": prep.is_synthetic,
        "regions_source": prep.regions_source,
    }
    return AnalysisResult(
        orders, customers, stores_view, regions, features, scored, candidates, coverage, hotspots, summary, params,
        notes,
    )
