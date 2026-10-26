"""Input validation for sales, store and region datasets.

Design rules
------------
* Nothing is silently invented. When a column is missing and we have to fall
  back (e.g. every row becomes its own customer), a warning is returned.
* Rows that cannot be used (bad coordinates, missing sales) are dropped and
  *counted*, so the user can see exactly what happened.
* This module only needs pandas/numpy. Geometry checks operate on whatever
  GeoDataFrame is passed in, so geopandas is not imported here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when an input dataset cannot be used. The message is user-facing."""


class MissingDependencyError(ImportError):
    """Raised when an optional package needed for a feature is not installed."""


# --------------------------------------------------------------------------
# Schema definition
# --------------------------------------------------------------------------
SALES_REQUIRED = ["latitude", "longitude", "sales"]
SALES_OPTIONAL = ["order_id", "customer_id", "date", "region", "category", "demand"]
STORES_REQUIRED = ["store_id", "latitude", "longitude"]
STORES_OPTIONAL = ["store_name", "store_type", "region"]

COLUMN_ALIASES = {
    "lat": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    "longitude": "longitude",
    "revenue": "sales",
    "amount": "sales",
    "sales_amount": "sales",
    "total_sales": "sales",
    "customerid": "customer_id",
    "cust_id": "customer_id",
    "customer": "customer_id",
    "storeid": "store_id",
    "store": "store_id",
    "orderid": "order_id",
    "order_date": "date",
    "product_category": "category",
    "type": "store_type",
}

STORE_TYPE_MAP = {
    "own": "own",
    "our": "own",
    "existing": "own",
    "company": "own",
    "branch": "own",
    "competitor": "competitor",
    "competition": "competitor",
    "rival": "competitor",
}

OUTSIDE_REGION = "Outside regions"


@dataclass
class ValidationResult:
    """Cleaned data plus a human-readable audit trail."""

    data: object  # pandas.DataFrame or geopandas.GeoDataFrame
    warnings: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, trim, replace spaces with '_' and apply known aliases."""
    out = df.copy()
    renamed = []
    for col in out.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        renamed.append(COLUMN_ALIASES.get(key, key))
    out.columns = renamed
    dupes = out.columns[out.columns.duplicated()].unique().tolist()
    if dupes:
        raise ValidationError(
            f"Duplicate column(s) after normalising names: {dupes}. "
            "Please keep only one column per field (e.g. either 'lat' or 'latitude')."
        )
    return out


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(
            f"{label}: missing required column(s) {missing}. "
            f"Columns found: {list(df.columns)}. "
            "See the 'Data & Upload' page for the expected schema."
        )


def validate_coordinates(
    df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce coordinates to numbers and drop unusable rows.

    A row is dropped if latitude/longitude is missing/non-numeric, outside
    [-90, 90] / [-180, 180], or exactly (0, 0) ("null island", almost always a
    placeholder for missing data).

    Returns the cleaned frame and a dict of counts per rejection reason.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValidationError(f"Coordinate columns '{lat_col}' and '{lon_col}' are required.")

    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    lat, lon = out[lat_col], out[lon_col]
    missing = lat.isna() | lon.isna() | ~np.isfinite(lat.to_numpy(float)) | ~np.isfinite(lon.to_numpy(float))
    out_of_range = ~missing & ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180))
    null_island = ~missing & ~out_of_range & (lat == 0) & (lon == 0)

    bad = missing | out_of_range | null_island
    counts = {
        "missing_or_non_numeric": int(missing.sum()),
        "out_of_range": int(out_of_range.sum()),
        "null_island_0_0": int(null_island.sum()),
    }
    return out.loc[~bad].reset_index(drop=True), counts


def _coord_warning(counts: dict[str, int], label: str) -> list[str]:
    total = sum(counts.values())
    if total == 0:
        return []
    detail = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v)
    return [f"{label}: dropped {total} row(s) with invalid coordinates ({detail})."]


def _parse_dates(col: pd.Series) -> pd.Series:
    """Parse dates tolerantly: ISO variants first, then mixed formats; unparseable -> NaT; tz stripped."""
    parsed = pd.to_datetime(col, errors="coerce", format="ISO8601")
    missing = parsed.isna() & col.notna()
    if missing.any():
        parsed = parsed.fillna(pd.to_datetime(col[missing], errors="coerce", format="mixed"))
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


# --------------------------------------------------------------------------
# Sales
# --------------------------------------------------------------------------
def validate_sales_df(df: pd.DataFrame | None) -> ValidationResult:
    """Validate and clean a sales/customer dataset.

    Required columns: latitude, longitude, sales.
    Optional: order_id, customer_id, date, region, category, demand.
    """
    if df is None or len(df) == 0:
        raise ValidationError("The sales dataset is empty. Upload a file with at least one row.")

    warnings: list[str] = []
    rows_in = len(df)
    out = normalize_columns(df)
    _require_columns(out, SALES_REQUIRED, "Sales data")

    # sales ---------------------------------------------------------------
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce")
    n_nan = int(out["sales"].isna().sum())
    if n_nan:
        warnings.append(f"Sales data: dropped {n_nan} row(s) with missing or non-numeric 'sales'.")
        out = out.loc[out["sales"].notna()]
    n_neg = int((out["sales"] < 0).sum())
    if n_neg:
        warnings.append(
            f"Sales data: dropped {n_neg} row(s) with negative 'sales' (refunds are not modelled)."
        )
        out = out.loc[out["sales"] >= 0]

    # coordinates ---------------------------------------------------------
    out, counts = validate_coordinates(out)
    warnings += _coord_warning(counts, "Sales data")

    # date ----------------------------------------------------------------
    if "date" in out.columns:
        parsed = _parse_dates(out["date"])
        if parsed.notna().sum() == 0 and len(out) > 0:
            warnings.append("Sales data: 'date' column could not be parsed and was ignored.")
            out = out.drop(columns="date")
        else:
            n_bad = int(parsed.isna().sum())
            if n_bad:
                warnings.append(f"Sales data: dropped {n_bad} row(s) with unparseable 'date'.")
            out["date"] = parsed
            out = out.loc[parsed.notna()]

    # identifiers / categories ---------------------------------------------
    if "customer_id" not in out.columns:
        out = out.reset_index(drop=True)
        out["customer_id"] = [f"row_{i}" for i in range(len(out))]
        warnings.append(
            "Sales data: no 'customer_id' column - every row is treated as a separate customer, "
            "so customer counts equal row counts."
        )
    else:
        out["customer_id"] = out["customer_id"].astype(str)

    if "category" in out.columns:
        out["category"] = out["category"].fillna("Unknown").astype(str)
    if "region" in out.columns:
        out["region"] = out["region"].fillna("Unknown").astype(str)

    # demand --------------------------------------------------------------
    if "demand" in out.columns:
        d = pd.to_numeric(out["demand"], errors="coerce")
        if d.isna().any() or (d < 0).any():
            warnings.append(
                "Sales data: 'demand' contains missing, non-numeric or negative values, so it was "
                "ignored. Sales revenue is used as a demand proxy instead."
            )
            out = out.drop(columns="demand")
        else:
            out["demand"] = d

    out = out.reset_index(drop=True)
    if out.empty:
        raise ValidationError(
            "No usable rows remain after validation. " + " ".join(warnings)
        )
    return ValidationResult(out, warnings, rows_in, len(out))


# --------------------------------------------------------------------------
# Stores
# --------------------------------------------------------------------------
def validate_stores_df(df: pd.DataFrame | None) -> ValidationResult:
    """Validate and clean a store dataset (own branches and/or competitors).

    Required: store_id, latitude, longitude. Optional: store_name, store_type, region.
    ``store_type`` must be 'own' or 'competitor' (a few synonyms are accepted).
    If it is absent, all stores are treated as OWN stores and a warning says so.
    """
    if df is None or len(df) == 0:
        raise ValidationError("The store dataset is empty. Upload a file with at least one row.")

    warnings: list[str] = []
    rows_in = len(df)
    out = normalize_columns(df)
    _require_columns(out, STORES_REQUIRED, "Store data")

    out, counts = validate_coordinates(out)
    warnings += _coord_warning(counts, "Store data")

    out["store_id"] = out["store_id"].astype(str)
    n_dupe = int(out["store_id"].duplicated().sum())
    if n_dupe:
        warnings.append(f"Store data: dropped {n_dupe} duplicate store_id row(s) (first kept).")
        out = out.drop_duplicates("store_id", keep="first")

    if "store_type" in out.columns:
        mapped = out["store_type"].astype(str).str.strip().str.lower().map(STORE_TYPE_MAP)
        unknown = sorted(out.loc[mapped.isna(), "store_type"].astype(str).unique().tolist())
        if unknown:
            raise ValidationError(
                f"Store data: unknown store_type value(s) {unknown}. "
                "Use 'own' for your branches and 'competitor' for competitors."
            )
        out["store_type"] = mapped
    else:
        out["store_type"] = "own"
        warnings.append(
            "Store data: no 'store_type' column - ALL stores are treated as your own branches "
            "(no competitor analysis possible)."
        )

    if "store_name" not in out.columns:
        out["store_name"] = out["store_id"]
    if "region" in out.columns:
        out["region"] = out["region"].fillna("Unknown").astype(str)

    out = out.reset_index(drop=True)
    if out.empty:
        raise ValidationError("No usable store rows remain after validation. " + " ".join(warnings))
    return ValidationResult(out, warnings, rows_in, len(out))


def empty_stores_frame() -> pd.DataFrame:
    """A schema-correct, zero-row store table (used when the user has no store data)."""
    return pd.DataFrame(
        {
            "store_id": pd.Series(dtype=str),
            "store_name": pd.Series(dtype=str),
            "store_type": pd.Series(dtype=str),
            "latitude": pd.Series(dtype=float),
            "longitude": pd.Series(dtype=float),
            "region": pd.Series(dtype=str),
        }
    )


# --------------------------------------------------------------------------
# Regions (polygons)
# --------------------------------------------------------------------------
REGION_ID_CANDIDATES = ["region", "region_id", "region_name", "name", "zone", "district", "id"]


def validate_regions_gdf(gdf) -> ValidationResult:
    """Validate an administrative-boundary GeoDataFrame and return it in EPSG:4326.

    * Needs polygon geometries and one identifying column (see REGION_ID_CANDIDATES).
    * Missing/empty geometries are dropped, invalid ones repaired with make_valid.
    * A missing CRS is only accepted when the coordinates clearly look like lon/lat.
    """
    if gdf is None or len(gdf) == 0:
        raise ValidationError("The boundary file contains no features.")
    if not hasattr(gdf, "geometry"):
        raise ValidationError(
            "Boundary files must be GeoJSON, a zipped Shapefile or GeoParquet (they need geometry)."
        )

    warnings: list[str] = []
    rows_in = len(gdf)
    out = gdf.copy()
    out.columns = [c if c == "geometry" else str(c).strip().lower().replace(" ", "_") for c in out.columns]

    bad_geom = out.geometry.isna() | out.geometry.is_empty
    if bad_geom.any():
        warnings.append(f"Boundaries: dropped {int(bad_geom.sum())} feature(s) with missing/empty geometry.")
        out = out.loc[~bad_geom]

    is_poly = out.geom_type.isin(["Polygon", "MultiPolygon"])
    if not is_poly.all():
        warnings.append(
            f"Boundaries: dropped {int((~is_poly).sum())} non-polygon feature(s) "
            "(boundaries must be Polygon or MultiPolygon)."
        )
        out = out.loc[is_poly]
    if out.empty:
        raise ValidationError("The boundary file has no polygon features.")

    id_col = next((c for c in REGION_ID_CANDIDATES if c in out.columns), None)
    if id_col is None:
        raise ValidationError(
            "Boundary file needs a column identifying each region, named one of "
            f"{REGION_ID_CANDIDATES}. Columns found: {[c for c in out.columns if c != 'geometry']}."
        )
    if id_col != "region":
        out = out.rename(columns={id_col: "region"})
        warnings.append(f"Boundaries: using column '{id_col}' as the region identifier.")
    out["region"] = out["region"].astype(str)

    # CRS ------------------------------------------------------------------
    if out.crs is None:
        minx, miny, maxx, maxy = out.total_bounds
        if -180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90:
            out = out.set_crs("EPSG:4326")
            warnings.append("Boundaries: no CRS was defined; coordinates look like lon/lat so EPSG:4326 was assumed.")
        else:
            raise ValidationError(
                "The boundary file has no coordinate reference system (CRS) and its coordinates are not "
                "lon/lat. Add the CRS (e.g. a .prj file inside the Shapefile zip) and upload again."
            )
    out = out.to_crs("EPSG:4326")

    invalid = ~out.geometry.is_valid
    if invalid.any():
        warnings.append(f"Boundaries: repaired {int(invalid.sum())} invalid geometry(ies) with make_valid.")
        out["geometry"] = out.geometry.make_valid()
        out = out.loc[out.geom_type.isin(["Polygon", "MultiPolygon"])]  # make_valid may emit collections

    if out["region"].duplicated().any():
        warnings.append("Boundaries: duplicate region ids were merged (dissolved) into single regions.")
        out = out[["region", "geometry"]].dissolve(by="region", as_index=False)

    out = out[["region", "geometry"]].reset_index(drop=True)
    if out.empty:
        raise ValidationError("No valid polygon features remain after validation.")
    return ValidationResult(out, warnings, rows_in, len(out))
