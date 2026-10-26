"""Tests for coordinate and schema validation (pandas only - no geopandas needed)."""
import numpy as np
import pandas as pd
import pytest

from src.validation import (
    ValidationError,
    normalize_columns,
    validate_coordinates,
    validate_sales_df,
    validate_stores_df,
)


def test_validate_coordinates_drops_bad_rows_and_counts_reasons():
    df = pd.DataFrame(
        {
            "latitude": [12.9, 95.0, "abc", None, 0.0, 13.1],
            "longitude": [77.5, 77.5, 77.5, 77.5, 0.0, 200.0],
        }
    )
    clean, counts = validate_coordinates(df)
    assert len(clean) == 1
    assert clean.iloc[0]["latitude"] == pytest.approx(12.9)
    assert counts == {"missing_or_non_numeric": 2, "out_of_range": 2, "null_island_0_0": 1}


def test_validate_coordinates_requires_columns():
    with pytest.raises(ValidationError, match="required"):
        validate_coordinates(pd.DataFrame({"lat": [1.0]}))


def test_normalize_columns_applies_aliases():
    df = pd.DataFrame({"Lat": [1], "LNG": [2], "Revenue": [3], "Customer ID": ["a"]})
    out = normalize_columns(df)
    assert list(out.columns) == ["latitude", "longitude", "sales", "customer_id"]


def test_normalize_columns_rejects_duplicates_after_aliasing():
    with pytest.raises(ValidationError, match="Duplicate"):
        normalize_columns(pd.DataFrame({"lat": [1], "latitude": [2]}))


def test_sales_missing_required_columns_message_lists_them():
    with pytest.raises(ValidationError) as err:
        validate_sales_df(pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}))
    assert "sales" in str(err.value)


@pytest.mark.parametrize("bad", [None, pd.DataFrame()])
def test_empty_sales_raises(bad):
    with pytest.raises(ValidationError, match="empty"):
        validate_sales_df(bad)


def test_sales_cleaning_and_warnings():
    df = pd.DataFrame(
        {
            "lat": [12.9, 12.95, 99.0, 12.8, 12.7],
            "lon": [77.5, 77.6, 77.5, 77.4, 77.3],
            "amount": [100, -5, 50, "x", 30],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "not a date"],
        }
    )
    res = validate_sales_df(df)
    # kept: row0 only (row1 negative, row2 bad lat, row3 bad sales, row4 bad date)
    assert len(res.data) == 1
    assert res.rows_dropped == 4
    assert pd.api.types.is_datetime64_any_dtype(res.data["date"])
    joined = " ".join(res.warnings)
    assert "negative" in joined and "invalid coordinates" in joined and "date" in joined
    # no customer_id supplied -> derived, and the user is told
    assert "customer_id" in res.data.columns
    assert any("customer_id" in w for w in res.warnings)


def test_all_rows_invalid_raises():
    df = pd.DataFrame({"latitude": [999.0], "longitude": [0.5], "sales": [10]})
    with pytest.raises(ValidationError, match="No usable rows"):
        validate_sales_df(df)


def test_bad_demand_column_is_ignored_with_warning():
    df = pd.DataFrame({"latitude": [12.9, 12.8], "longitude": [77.5, 77.4], "sales": [1, 2], "demand": [3, None]})
    res = validate_sales_df(df)
    assert "demand" not in res.data.columns
    assert any("demand" in w for w in res.warnings)


def test_stores_type_normalisation_and_duplicates():
    df = pd.DataFrame(
        {
            "store_id": ["a", "b", "b", "c"],
            "latitude": [12.9, 12.8, 12.8, 12.7],
            "longitude": [77.5, 77.4, 77.4, 77.3],
            "store_type": ["Own", "Rival", "Rival", "branch"],
        }
    )
    res = validate_stores_df(df)
    assert sorted(res.data["store_type"].unique()) == ["competitor", "own"]
    assert len(res.data) == 3
    assert any("duplicate" in w for w in res.warnings)


def test_stores_unknown_type_raises_with_values():
    df = pd.DataFrame({"store_id": ["a"], "latitude": [1.0], "longitude": [2.0], "store_type": ["kiosk"]})
    with pytest.raises(ValidationError, match="kiosk"):
        validate_stores_df(df)


def test_stores_without_type_default_to_own_with_warning():
    df = pd.DataFrame({"store_id": ["a"], "latitude": [1.0], "longitude": [2.0]})
    res = validate_stores_df(df)
    assert (res.data["store_type"] == "own").all()
    assert any("store_type" in w for w in res.warnings)


def test_mixed_date_formats_are_parsed_not_dropped():
    df = pd.DataFrame(
        {
            "latitude": [12.9, 12.8, 12.7],
            "longitude": [77.5, 77.4, 77.3],
            "sales": [1, 2, 3],
            "date": ["2025-01-05", "2025-01-06 14:30:00", "07 Jan 2025"],
        }
    )
    res = validate_sales_df(df)
    assert len(res.data) == 3
    assert res.data["date"].dt.day.tolist() == [5, 6, 7]
