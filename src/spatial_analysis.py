"""Spatial operations: joins, containment, intersection, nearest neighbour,
buffers, coverage and DBSCAN hotspots.

CRS policy
----------
Coordinates arrive as WGS84 lon/lat (EPSG:4326). Degrees are NOT metres, so
every distance, buffer or area is computed after reprojecting to a metric CRS -
by default the UTM zone of the data's centre (see :func:`utm_epsg`). Results
that are drawn on maps are converted back to EPSG:4326.

geopandas / scikit-learn / scipy are imported lazily so the rest of the app
(and the pure-pandas tests) still work if they are missing.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .validation import OUTSIDE_REGION, MissingDependencyError, ValidationError

WGS84 = "EPSG:4326"
EARTH_RADIUS_KM = 6371.0088
MAX_GRID_CELLS = 400


# --------------------------------------------------------------------------
# Lazy imports
# --------------------------------------------------------------------------
def _gpd():
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError(
            "geopandas is required for spatial analysis. Install it with: pip install geopandas"
        ) from exc
    return gpd


# --------------------------------------------------------------------------
# CRS helpers and plain-math distance
# --------------------------------------------------------------------------
def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84/UTM zone containing (lon, lat). e.g. (77.6, 13.0) -> 32643."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (works with scalars or numpy arrays)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def choose_projected_crs(gdf) -> str:
    """Pick the UTM CRS for the centre of ``gdf`` (assumed to be EPSG:4326)."""
    minx, miny, maxx, maxy = gdf.to_crs(WGS84).total_bounds
    # NOTE: if the data spans several UTM zones (> ~12 degrees of longitude) a single
    # UTM zone distorts distances; pass an explicit regional CRS to the functions instead.
    return f"EPSG:{utm_epsg((minx + maxx) / 2, (miny + maxy) / 2)}"


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def points_to_gdf(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"):
    """DataFrame with lat/lon columns -> point GeoDataFrame in EPSG:4326."""
    gpd = _gpd()
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValidationError(f"Columns '{lat_col}' and '{lon_col}' are required to build points.")
    base = df.reset_index(drop=True).copy()
    return gpd.GeoDataFrame(base, geometry=gpd.points_from_xy(base[lon_col], base[lat_col]), crs=WGS84)


def make_grid_regions(points_gdf, cell_km: float = 5.0, padding_km: float = 1.0, prefix: str = "G"):
    """Build square grid cells covering ``points_gdf`` (fallback when no boundaries are supplied).

    The grid is NOT an administrative boundary: it is an analysis convenience.
    Cells with no points are discarded. Cell size grows automatically if the
    grid would exceed MAX_GRID_CELLS.
    """
    gpd = _gpd()
    from shapely.geometry import box

    if len(points_gdf) == 0:
        raise ValidationError("Cannot build a grid without any points.")
    crs = choose_projected_crs(points_gdf)
    proj = points_gdf.to_crs(crs)
    minx, miny, maxx, maxy = proj.total_bounds
    pad = padding_km * 1000.0
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

    size = max(cell_km, 0.1) * 1000.0
    while math.ceil((maxx - minx) / size) * math.ceil((maxy - miny) / size) > MAX_GRID_CELLS:
        size *= 1.25
    nx, ny = math.ceil((maxx - minx) / size), math.ceil((maxy - miny) / size)

    cells, ids = [], []
    for r in range(ny):
        for c in range(nx):
            cells.append(box(minx + c * size, miny + r * size, minx + (c + 1) * size, miny + (r + 1) * size))
            ids.append(f"{prefix}{r + 1:02d}-{c + 1:02d}")
    grid = gpd.GeoDataFrame({"region": ids}, geometry=cells, crs=crs)

    hit = gpd.sjoin(proj[["geometry"]], grid, how="inner", predicate="within")
    grid = grid[grid["region"].isin(hit["region"].unique())].reset_index(drop=True)
    return grid.to_crs(WGS84)


# --------------------------------------------------------------------------
# Spatial join / containment / intersection
# --------------------------------------------------------------------------
def assign_regions(points_gdf, regions_gdf, region_col: str = "region", predicate: str = "within"):
    """Spatial join: tag each point with the region polygon that contains it.

    * Points outside every polygon get ``OUTSIDE_REGION``.
    * Points on a shared boundary match several polygons; the first match is kept.
    * An existing ``region`` column on the points is preserved as ``region_input``.
    """
    gpd = _gpd()
    pts = points_gdf.reset_index(drop=True).copy()
    if region_col in pts.columns:
        pts = pts.rename(columns={region_col: f"{region_col}_input"})
    reg = regions_gdf[[region_col, "geometry"]].to_crs(pts.crs)

    joined = gpd.sjoin(pts, reg, how="left", predicate=predicate)
    joined = joined[~joined.index.duplicated(keep="first")].drop(columns=["index_right"], errors="ignore")
    joined[region_col] = joined[region_col].fillna(OUTSIDE_REGION).astype(str)
    return joined.sort_index()


def containment_counts(points_gdf, regions_gdf, region_col: str = "region") -> pd.DataFrame:
    """Point-in-polygon counts: how many points fall inside each region."""
    joined = assign_regions(points_gdf, regions_gdf, region_col)
    counts = joined.groupby(region_col).size().rename("n_points")
    all_regions = pd.Index(regions_gdf[region_col].astype(str).unique(), name=region_col)
    return counts.reindex(all_regions, fill_value=0).reset_index()


def points_within_polygon(points_gdf, polygon):
    """Explicit point-in-polygon filter. ``polygon`` is a shapely geometry in the points' CRS."""
    return points_gdf[points_gdf.geometry.within(polygon)]


def intersection_analysis(layer_a, layer_b, crs: str | None = None):
    """Geometric intersection of two polygon layers (``geopandas.overlay``).

    Returns the intersection pieces in EPSG:4326 plus ``area_km2`` computed in a
    projected CRS. Attribute columns of both layers are kept (name clashes get
    ``_1`` / ``_2`` suffixes from geopandas).
    """
    gpd = _gpd()
    crs = crs or choose_projected_crs(layer_a)
    a, b = layer_a.to_crs(crs), layer_b.to_crs(crs)
    pieces = gpd.overlay(a, b, how="intersection", keep_geom_type=True)
    pieces["area_km2"] = pieces.geometry.area / 1e6
    return pieces.to_crs(WGS84)


# --------------------------------------------------------------------------
# Nearest neighbour
# --------------------------------------------------------------------------
def nearest_distances_m(src_xy: np.ndarray, tgt_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each source point return (distance, index) of the nearest target. Inputs are projected metres."""
    if len(tgt_xy) == 0:
        raise ValidationError("No target points to measure distance to.")
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("scipy is required for nearest-neighbour search.") from exc
    dist, idx = cKDTree(np.asarray(tgt_xy, float)).query(np.asarray(src_xy, float), k=1)
    return dist, idx


def add_nearest_store(points_gdf, stores_gdf, crs: str | None = None):
    """Add ``nearest_store_id`` and ``nearest_store_dist_km`` to ``points_gdf``.

    Distances are Euclidean in a projected CRS (metres -> km).
    """
    crs = crs or choose_projected_crs(points_gdf)
    p, s = points_gdf.to_crs(crs), stores_gdf.reset_index(drop=True).to_crs(crs)
    src = np.column_stack([p.geometry.x.to_numpy(), p.geometry.y.to_numpy()])
    tgt = np.column_stack([s.geometry.x.to_numpy(), s.geometry.y.to_numpy()])
    dist, idx = nearest_distances_m(src, tgt)
    out = points_gdf.copy()
    out["nearest_store_id"] = s["store_id"].to_numpy()[idx]
    out["nearest_store_dist_km"] = dist / 1000.0
    return out


# --------------------------------------------------------------------------
# Buffers and coverage
# --------------------------------------------------------------------------
def store_buffers(stores_gdf, radius_km: float, crs: str | None = None):
    """One circular service area per store (buffer computed in metres), returned in EPSG:4326."""
    if radius_km <= 0:
        raise ValidationError("Coverage radius must be greater than zero.")
    crs = crs or choose_projected_crs(stores_gdf)
    proj = stores_gdf.to_crs(crs).copy()
    proj["geometry"] = proj.geometry.buffer(radius_km * 1000.0, resolution=32)
    return proj.to_crs(WGS84)


def coverage_union(stores_gdf, radius_km: float, crs: str | None = None):
    """Single dissolved coverage polygon of all store buffers (1-row GeoDataFrame, EPSG:4326)."""
    gpd = _gpd()
    buffers = store_buffers(stores_gdf, radius_km, crs)
    union = buffers.geometry.union_all()
    return gpd.GeoDataFrame({"coverage_km": [radius_km]}, geometry=[union], crs=WGS84)


def points_in_coverage(points_gdf, coverage_gdf) -> pd.Series:
    """Point-in-polygon test of each point against the dissolved coverage area (boolean Series)."""
    poly = coverage_gdf.to_crs(points_gdf.crs).geometry.iloc[0]
    return points_gdf.geometry.within(poly)


def coverage_area_by_region(regions_gdf, coverage_gdf, crs: str | None = None) -> pd.DataFrame:
    """Share of each region's AREA that lies inside the coverage polygon (uses intersection overlay)."""
    crs = crs or choose_projected_crs(regions_gdf)
    reg = regions_gdf[["region", "geometry"]].reset_index(drop=True)
    area = reg.to_crs(crs).geometry.area / 1e6
    base = pd.DataFrame({"region": reg["region"].astype(str), "area_km2": area.to_numpy()})

    pieces = intersection_analysis(reg, coverage_gdf[["geometry"]], crs=crs)
    if len(pieces):
        covered = pieces.groupby(pieces["region"].astype(str))["area_km2"].sum().rename("covered_area_km2")
        base = base.merge(covered, left_on="region", right_index=True, how="left")
    else:
        base["covered_area_km2"] = 0.0
    base["covered_area_km2"] = base["covered_area_km2"].fillna(0.0)
    base["covered_area_share"] = (base["covered_area_km2"] / base["area_km2"]).clip(0, 1)
    return base


# --------------------------------------------------------------------------
# DBSCAN hotspots
# --------------------------------------------------------------------------
def dbscan_hotspots(
    customers: pd.DataFrame,
    eps_km: float = 2.0,
    min_samples: int = 15,
    value_col: str | None = "sales",
):
    """Density-based clustering of customer locations (DBSCAN on projected metres).

    Returns ``(labels, summary)``. ``labels`` aligns with ``customers``
    (-1 = noise / not in a hotspot). ``summary`` has one row per hotspot with
    size, total value and a (value-weighted) centre.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("scikit-learn is required for DBSCAN hotspots.") from exc

    if len(customers) == 0:
        return np.array([], dtype=int), pd.DataFrame()
    gdf = points_to_gdf(customers)
    proj = gdf.to_crs(choose_projected_crs(gdf))
    xy = np.column_stack([proj.geometry.x.to_numpy(), proj.geometry.y.to_numpy()])
    labels = DBSCAN(eps=eps_km * 1000.0, min_samples=int(min_samples)).fit_predict(xy)
    return labels, summarize_hotspots(customers, labels, value_col)


def summarize_hotspots(customers: pd.DataFrame, labels, value_col: str | None = "sales") -> pd.DataFrame:
    """Aggregate DBSCAN labels into a hotspot table (pure pandas)."""
    df = customers.reset_index(drop=True).copy()
    df["hotspot"] = np.asarray(labels)
    df = df[df["hotspot"] >= 0]
    if df.empty:
        return pd.DataFrame(columns=["hotspot", "n_customers", "value", "latitude", "longitude"])
    rows = []
    for h, g in df.groupby("hotspot"):
        w = g[value_col].to_numpy(float) if value_col and value_col in g.columns else np.ones(len(g))
        if w.sum() <= 0:
            w = np.ones(len(g))
        rows.append(
            {
                "hotspot": int(h),
                "n_customers": int(len(g)),
                "value": float(g[value_col].sum()) if value_col and value_col in g.columns else float(len(g)),
                "latitude": float(np.average(g["latitude"], weights=w)),
                "longitude": float(np.average(g["longitude"], weights=w)),
            }
        )
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
