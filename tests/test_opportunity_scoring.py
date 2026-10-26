"""Tests for opportunity scoring, normalisation, candidate sites and pipeline helpers (pandas only)."""
import numpy as np
import pandas as pd
import pytest

from src import opportunity_scoring as osc
from src.pipeline import Filters, apply_filters, build_customers


def _customers():
    """Region HOT: lots of demand, far from stores. COLD: little demand, close to a store. MID: in between."""
    rows = []
    for region, n, sales, dist in [("HOT", 40, 100.0, 9.0), ("MID", 20, 60.0, 4.0), ("COLD", 10, 20.0, 1.0)]:
        for i in range(n):
            rows.append(
                {
                    "customer_id": f"{region}{i}", "region": region, "sales": sales, "orders": 2,
                    "nearest_store_dist_km": dist, "covered": dist <= 5.0,
                    "latitude": 13.0 + (0.1 if region == "HOT" else 0.0) + i * 1e-4, "longitude": 77.5 + i * 1e-4,
                }
            )
    return pd.DataFrame(rows)


def _stores():
    return pd.DataFrame(
        {
            "region": ["COLD", "MID", "MID", "HOT"],
            "store_type": ["own", "own", "competitor", "competitor"],
        }
    )


def _features():
    return osc.build_region_features(_customers(), _stores(), ["HOT", "MID", "COLD"])


# ---------------------------------------------------------------- normalisation
def test_minmax_scales_to_unit_interval():
    out = osc.minmax(pd.Series([10, 20, 30]))
    assert out.tolist() == [0.0, 0.5, 1.0]
    assert osc.minmax(pd.Series([10, 20, 30]), invert=True).tolist() == [1.0, 0.5, 0.0]


def test_minmax_constant_series_is_neutral():
    assert osc.minmax(pd.Series([7, 7, 7])).tolist() == [0.5, 0.5, 0.5]


def test_normalize_weights_rescales_and_restricts():
    w = osc.normalize_weights({"demand": 2, "store_gap": 2, "distance": 4, "competition": 2})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["distance"] == pytest.approx(0.4)
    only = osc.normalize_weights({"demand": 1, "store_gap": 1, "distance": 1, "competition": 1}, ["demand", "competition"])
    assert only == {"demand": 0.5, "competition": 0.5}


def test_normalize_weights_edge_cases():
    with pytest.raises(ValueError):
        osc.normalize_weights({"demand": -1})
    zero = osc.normalize_weights({"demand": 0, "store_gap": 0, "distance": 0, "competition": 0})
    assert all(v == pytest.approx(0.25) for v in zero.values())


# ---------------------------------------------------------------- features
def test_region_features_counts_and_aggregates():
    f = _features().set_index("region")
    assert f.loc["HOT", "customers"] == 40 and f.loc["HOT", "sales"] == pytest.approx(4000)
    assert f.loc["MID", ["own_stores", "competitors"]].tolist() == [1, 1]
    assert f.loc["HOT", "own_stores"] == 0 and f.loc["HOT", "competitors"] == 1
    assert f.loc["HOT", "mean_dist_km"] == pytest.approx(9.0)
    assert f.loc["COLD", "uncovered_share"] == pytest.approx(0.0)
    assert f.loc["HOT", "uncovered_share"] == pytest.approx(1.0)
    assert (f["demand_basis"] == "sales (proxy for demand)").all()


def test_outside_region_is_excluded_from_features():
    f = osc.build_region_features(_customers(), _stores(), ["HOT", "Outside regions"])
    assert f["region"].tolist() == ["HOT"]


# ---------------------------------------------------------------- scoring
def test_high_demand_underserved_region_ranks_first():
    scored = osc.score_regions(_features())
    assert scored.iloc[0]["region"] == "HOT"
    assert scored.iloc[-1]["region"] == "COLD"
    assert scored["rank"].tolist() == [1, 2, 3]
    assert scored["opportunity_score"].is_monotonic_decreasing
    assert scored["opportunity_score"].between(0, 100).all()


def test_scores_equal_weighted_sum_of_components():
    weights = {"demand": 0.4, "store_gap": 0.2, "distance": 0.2, "competition": 0.2}
    scored = osc.score_regions(_features(), weights)
    for _, r in scored.iterrows():
        expected = 100 * (0.4 * r["c_demand"] + 0.2 * r["c_store_gap"] + 0.2 * r["c_distance"] + 0.2 * r["c_competition"])
        assert r["opportunity_score"] == pytest.approx(expected, abs=0.01)


def test_changing_weights_changes_ranking():
    demand_only = osc.score_regions(_features(), {"demand": 1, "store_gap": 0, "distance": 0, "competition": 0})
    comp_only = osc.score_regions(_features(), {"demand": 0, "store_gap": 0, "distance": 0, "competition": 1})
    assert demand_only.iloc[0]["region"] == "HOT"
    # COLD has no competitor stores -> best on the competition component alone
    assert comp_only.iloc[0]["region"] == "COLD"


def test_missing_competitor_data_excludes_component_instead_of_inventing_it():
    f = osc.build_region_features(_customers(), _stores()[_stores()["store_type"] == "own"], ["HOT", "MID", "COLD"])
    scored = osc.score_regions(f, has_competitors=False)
    assert "competition" not in scored["components_used"].iloc[0]
    assert (scored["pts_competition"] == 0).all()
    assert (scored["opportunity_score"] <= 100).all()


def test_no_own_stores_leaves_only_demand_and_competition():
    f = osc.build_region_features(_customers(), _stores()[_stores()["store_type"] == "competitor"], ["HOT", "MID", "COLD"])
    scored = osc.score_regions(f, has_own_stores=False, has_competitors=True)
    assert scored["components_used"].iloc[0] == "demand, competition"
    assert scored["c_store_gap"].isna().all()


def test_explanations_are_present_and_reference_measured_values():
    scored = osc.score_regions(_features())
    top = scored.iloc[0]
    assert top["region"] == "HOT"
    assert "Strongest drivers" in top["explanation"] and "0 own store(s)" in top["explanation"]
    assert scored["explanation"].str.len().min() > 20


def test_opportunity_flag_requires_high_demand_and_low_presence():
    scored = osc.score_regions(_features())
    flags = scored.set_index("region")["high_demand_low_presence"]
    assert bool(flags["HOT"]) is True
    assert bool(flags["COLD"]) is False


def test_demand_column_used_when_available():
    c = _customers()
    c["demand"] = 1.0
    f = osc.build_region_features(c, _stores(), ["HOT", "MID", "COLD"]).set_index("region")
    assert f.loc["HOT", "demand_total"] == pytest.approx(40)
    assert f.loc["HOT", "demand_basis"] == "demand column"


def test_score_regions_rejects_empty_input():
    with pytest.raises(ValueError):
        osc.score_regions(pd.DataFrame())


# ---------------------------------------------------------------- candidate sites
def test_candidate_site_sits_inside_the_uncovered_customers_extent():
    cust = _customers()
    scored = osc.score_regions(osc.build_region_features(cust, _stores(), ["HOT", "MID", "COLD"]))
    cand = osc.suggest_candidate_sites(cust, scored, top_n=2)
    assert cand["region"].tolist() == ["HOT", "MID"] or cand["region"].iloc[0] == "HOT"
    hot = cust[cust["region"] == "HOT"]
    row = cand.iloc[0]
    assert hot["latitude"].min() <= row["latitude"] <= hot["latitude"].max()
    assert hot["longitude"].min() <= row["longitude"] <= hot["longitude"].max()
    assert "estimate" in row["location_basis"]


# ---------------------------------------------------------------- pipeline helpers
def _orders():
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["a", "a", "b", "c"],
            "date": pd.to_datetime(["2025-01-05", "2025-02-10", "2025-03-01", "2025-03-31 18:00"], format="ISO8601"),
            "region": ["R1", "R1", "R2", "R2"],
            "category": ["x", "y", "x", "x"],
            "sales": [10.0, 20.0, 5.0, 50.0],
            "latitude": [1.0, 1.0, 2.0, 3.0],
            "longitude": [1.0, 1.0, 2.0, 3.0],
            "nearest_store_dist_km": [1.0, 1.0, 2.0, 3.0],
            "nearest_store_id": ["s", "s", "s", "t"],
        }
    )


def test_apply_filters_date_region_category_sales():
    o = _orders()
    assert len(apply_filters(o, Filters())) == 4
    assert len(apply_filters(o, Filters(date_range=("2025-02-01", "2025-03-31")))) == 3  # end date inclusive of the whole day
    assert apply_filters(o, Filters(regions=["R2"]))["order_id"].tolist() == ["o3", "o4"]
    assert apply_filters(o, Filters(categories=["y"]))["order_id"].tolist() == ["o2"]
    assert apply_filters(o, Filters(sales_range=(10, 30)))["order_id"].tolist() == ["o1", "o2"]


def test_build_customers_aggregates_orders():
    c = build_customers(_orders()).set_index("customer_id")
    assert len(c) == 3
    assert c.loc["a", "sales"] == pytest.approx(30.0) and c.loc["a", "orders"] == 2
