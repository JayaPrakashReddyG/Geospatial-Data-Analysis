"""Tests for spatial joins, distances, buffers, intersections and clustering.

Pure-math tests always run. Tests that need geopandas are skipped when it is
not installed (``pytest.importorskip``).
"""
import numpy as np
import pandas as pd
import pytest

from src import spatial_analysis as sa

# A 0.1 x 0.1 degree square near 13N / 77.5E, split into two "regions" A (west) and B (east).
LON0, LAT0, STEP = 77.50, 13.00, 0.05


# ---------------------------------------------------------------- pure math
def test_utm_epsg_known_zones():
    assert sa.utm_epsg(77.6, 13.0) == 32643      # Bangalore area, 43N
    assert sa.utm_epsg(-74.0, 40.7) == 32618     # New York, 18N
    assert sa.utm_epsg(151.2, -33.9) == 32756    # Sydney, 56S


def test_haversine_known_distances():
    # 1 degree of latitude is ~111.2 km
    assert sa.haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.1)
    # Paris -> London ~ 343.5 km
    assert sa.haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, abs=2)
    assert sa.haversine_km(10, 20, 10, 20) == pytest.approx(0.0)


def test_degrees_are_not_metres():
    """One degree of longitude shrinks with latitude - why we must project before measuring."""
    at_equator = sa.haversine_km(0, 0, 0, 1)
    at_60n = sa.haversine_km(60, 0, 60, 1)
    assert at_60n == pytest.approx(at_equator / 2, rel=0.01)


def test_nearest_distances_m_basic():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 100.0]])
    tgt = np.array([[1.0, 0.0], [100.0, 90.0]])
    dist, idx = sa.nearest_distances_m(src, tgt)
    assert idx.tolist() == [0, 0, 1]
    assert dist.tolist() == pytest.approx([1.0, 9.0, 10.0])


def test_nearest_distances_requires_targets():
    with pytest.raises(Exception, match="No target"):
        sa.nearest_distances_m(np.array([[0.0, 0.0]]), np.empty((0, 2)))


def test_summarize_hotspots_ignores_noise_and_weights_centres():
    cust = pd.DataFrame(
        {"latitude": [1.0, 1.0, 5.0, 9.0], "longitude": [1.0, 3.0, 5.0, 9.0], "sales": [1.0, 3.0, 2.0, 2.0]}
    )
    out = sa.summarize_hotspots(cust, [0, 0, 1, -1])
    assert len(out) == 2
    top = out.iloc[0]  # hotspot 0 has the larger total value (4 vs 2)
    assert top["hotspot"] == 0 and top["n_customers"] == 2
    assert top["longitude"] == pytest.approx(2.5)  # (1*1 + 3*3)/4


# ---------------------------------------------------------------- geopandas
@pytest.fixture(scope="module")
def gpd():
    return pytest.importorskip("geopandas")


@pytest.fixture()
def regions(gpd):
    from shapely.geometry import box

    a = box(LON0, LAT0, LON0 + STEP, LAT0 + 2 * STEP)
    b = box(LON0 + STEP, LAT0, LON0 + 2 * STEP, LAT0 + 2 * STEP)
    return gpd.GeoDataFrame({"region": ["A", "B"]}, geometry=[a, b], crs="EPSG:4326")


def _points(rows):
    return sa.points_to_gdf(pd.DataFrame(rows, columns=["pid", "latitude", "longitude"]))


def test_spatial_join_assigns_regions_and_flags_outside(regions):
    pts = _points([("p1", 13.02, 77.52), ("p2", 13.02, 77.58), ("p3", 14.0, 78.5)])
    joined = sa.assign_regions(pts, regions)
    assert joined.set_index("pid")["region"].to_dict() == {"p1": "A", "p2": "B", "p3": "Outside regions"}
    assert len(joined) == 3  # no duplicate rows


def test_spatial_join_boundary_point_not_duplicated(regions):
    on_border = _points([("edge", 13.02, LON0 + STEP)])
    joined = sa.assign_regions(on_border, regions, predicate="intersects")
    assert len(joined) == 1


def test_existing_region_column_is_preserved_as_region_input(regions):
    pts = sa.points_to_gdf(pd.DataFrame({"latitude": [13.02], "longitude": [77.52], "region": ["legacy"]}))
    joined = sa.assign_regions(pts, regions)
    assert joined.loc[0, "region_input"] == "legacy" and joined.loc[0, "region"] == "A"


def test_containment_counts_include_empty_regions(regions):
    pts = _points([("p1", 13.02, 77.52), ("p2", 13.03, 77.53)])
    counts = sa.containment_counts(pts, regions).set_index("region")["n_points"].to_dict()
    assert counts == {"A": 2, "B": 0}


def test_point_in_polygon_filter(regions):
    pts = _points([("in", 13.02, 77.52), ("out", 13.02, 77.58)])
    inside = sa.points_within_polygon(pts, regions.geometry.iloc[0])
    assert inside["pid"].tolist() == ["in"]


def test_projected_distance_matches_haversine(gpd):
    a = _points([("a", 13.00, 77.50)])
    b = _points([("s1", 13.10, 77.60)])
    out = sa.add_nearest_store(a, b.rename(columns={"pid": "store_id"}))
    expected = float(sa.haversine_km(13.00, 77.50, 13.10, 77.60))
    assert out.loc[0, "nearest_store_dist_km"] == pytest.approx(expected, rel=0.005)
    assert out.loc[0, "nearest_store_id"] == "s1"


def test_nearest_store_picks_closest(gpd):
    custs = _points([("c1", 13.00, 77.50), ("c2", 13.20, 77.70)])
    stores = pd.DataFrame({"store_id": ["near_c1", "near_c2"], "latitude": [13.001, 13.199], "longitude": [77.501, 77.699]})
    out = sa.add_nearest_store(custs, sa.points_to_gdf(stores))
    assert out["nearest_store_id"].tolist() == ["near_c1", "near_c2"]
    assert (out["nearest_store_dist_km"] < 0.3).all()


def test_buffer_radius_is_in_metres_not_degrees(gpd):
    stores = sa.points_to_gdf(pd.DataFrame({"store_id": ["s"], "latitude": [13.0], "longitude": [77.5]}))
    cov = sa.coverage_union(stores, radius_km=5)
    crs = sa.choose_projected_crs(cov)
    area_km2 = cov.to_crs(crs).geometry.area.iloc[0] / 1e6
    assert area_km2 == pytest.approx(np.pi * 25, rel=0.01)  # pi r^2 with r = 5 km


def test_coverage_flags_points_inside_and_outside(gpd):
    stores = sa.points_to_gdf(pd.DataFrame({"store_id": ["s"], "latitude": [13.0], "longitude": [77.5]}))
    cov = sa.coverage_union(stores, radius_km=5)
    pts = _points([("close", 13.01, 77.51), ("far", 13.20, 77.70)])  # ~1.5 km and ~30 km away
    assert sa.points_in_coverage(pts, cov).tolist() == [True, False]


def test_intersection_and_covered_area_share(gpd, regions):
    # a store at the A/B border with a 2 km radius: covers parts of both regions
    store = sa.points_to_gdf(pd.DataFrame({"store_id": ["s"], "latitude": [13.05], "longitude": [LON0 + STEP]}))
    cov = sa.coverage_union(store, radius_km=2)
    pieces = sa.intersection_analysis(regions, cov)
    assert set(pieces["region"]) == {"A", "B"}
    shares = sa.coverage_area_by_region(regions, cov).set_index("region")
    assert 0 < shares.loc["A", "covered_area_share"] < 0.2
    assert shares.loc["A", "covered_area_km2"] == pytest.approx(shares.loc["B", "covered_area_km2"], rel=0.02)
    total = pieces["area_km2"].sum()
    assert total == pytest.approx(np.pi * 4, rel=0.02)


def test_make_grid_regions_cover_points(gpd):
    rng = np.random.default_rng(0)
    pts = sa.points_to_gdf(pd.DataFrame({"latitude": 13 + rng.random(50) * 0.2, "longitude": 77.5 + rng.random(50) * 0.2}))
    grid = sa.make_grid_regions(pts, cell_km=5)
    joined = sa.assign_regions(pts, grid)
    assert (joined["region"] != "Outside regions").all()


def test_dbscan_finds_two_clusters_and_noise(gpd):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    c1 = rng.normal([13.00, 77.50], 0.002, size=(30, 2))
    c2 = rng.normal([13.10, 77.60], 0.002, size=(30, 2))
    noise = np.array([[13.05, 77.80]])
    arr = np.vstack([c1, c2, noise])
    df = pd.DataFrame({"latitude": arr[:, 0], "longitude": arr[:, 1], "sales": 1.0})
    labels, summary = sa.dbscan_hotspots(df, eps_km=1.0, min_samples=5)
    assert len(summary) == 2
    assert labels[-1] == -1
