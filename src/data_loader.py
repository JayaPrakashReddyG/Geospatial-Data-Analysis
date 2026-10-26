"""Reading and writing tabular and geospatial files.

Supported input : CSV, XLSX, GeoJSON, zipped Shapefile, GeoParquet (and plain Parquet).
Supported output: CSV, XLSX, GeoJSON, zipped Shapefile, GeoParquet.

``source`` may be a filesystem path or a file-like object such as a Streamlit
``UploadedFile`` (anything with ``.read()`` and optionally ``.name``).
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from .validation import MissingDependencyError, ValidationError, validate_regions_gdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TABLE_EXTENSIONS = [".csv", ".xlsx", ".geojson", ".json", ".parquet", ".geoparquet", ".zip"]
BOUNDARY_EXTENSIONS = [".geojson", ".json", ".zip", ".parquet", ".geoparquet"]
MAX_UNZIPPED_BYTES = 300 * 1024 * 1024


def _gpd():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise MissingDependencyError(
            "geopandas is required to read this file type. Install it with: pip install geopandas"
        ) from exc
    return gpd


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
def load_sample_sales() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_sales.csv")


def load_sample_stores() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sample_stores.csv")


def load_sample_regions():
    """Synthetic zone polygons shipped with the project (already validated)."""
    gdf = _gpd().read_file(DATA_DIR / "sample_regions.geojson")
    return validate_regions_gdf(gdf).data


# --------------------------------------------------------------------------
# Generic reader
# --------------------------------------------------------------------------
def _name_and_bytes(source, filename: str | None) -> tuple[str, bytes | None, Path | None]:
    if hasattr(source, "read"):
        name = filename or getattr(source, "name", "") or ""
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return name, data, None
    path = Path(source)
    return filename or path.name, None, path


def read_any(source, filename: str | None = None):
    """Read a supported file into a DataFrame or GeoDataFrame.

    Raises ValidationError with an understandable message for unsupported or
    unreadable files.
    """
    name, data, path = _name_and_bytes(source, filename)
    ext = Path(name).suffix.lower()
    if ext not in TABLE_EXTENSIONS + [".shp"]:
        raise ValidationError(
            f"Unsupported file type '{ext or name}'. Supported: CSV, XLSX, GeoJSON, zipped Shapefile (.zip), GeoParquet."
        )

    def stream() -> io.BytesIO | Path:
        return io.BytesIO(data) if data is not None else path

    try:
        if ext == ".csv":
            try:
                return pd.read_csv(stream())
            except UnicodeDecodeError:
                return pd.read_csv(stream(), encoding="latin-1")
        if ext == ".xlsx":
            return pd.read_excel(stream(), engine="openpyxl")
        if ext in (".geojson", ".json"):
            return _gpd().read_file(stream())
        if ext in (".parquet", ".geoparquet"):
            gpd = _gpd()
            try:
                return gpd.read_parquet(stream())
            except (ValueError, KeyError):  # no GeoParquet metadata -> plain table
                return pd.read_parquet(stream())
        if ext == ".zip":
            return _read_zipped_shapefile(data if data is not None else path.read_bytes())
        if ext == ".shp":
            if path is None:
                raise ValidationError("Upload Shapefiles as a .zip containing the .shp, .shx, .dbf and .prj files.")
            return _gpd().read_file(path)
    except (ValidationError, MissingDependencyError):
        raise
    except pd.errors.EmptyDataError as exc:
        raise ValidationError(f"'{name}' is empty.") from exc
    except ImportError as exc:
        raise MissingDependencyError(f"A package needed to read '{name}' is missing: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface any parser error as a friendly message
        raise ValidationError(f"Could not read '{name}': {exc}") from exc
    raise ValidationError(f"Unsupported file type '{ext}'.")


def _read_zipped_shapefile(data: bytes):
    gpd = _gpd()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED_BYTES:
                raise ValidationError("The zip file is too large when unpacked.")
            for member in zf.infolist():  # guard against path traversal ("zip slip")
                target = (root / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValidationError("The zip file contains unsafe paths.")
            zf.extractall(root)
        shp = sorted(root.rglob("*.shp"))
        if not shp:
            raise ValidationError("No .shp file found inside the zip archive.")
        return gpd.read_file(shp[0])


# --------------------------------------------------------------------------
# Typed loaders
# --------------------------------------------------------------------------
def load_points_table(source, filename: str | None = None) -> pd.DataFrame:
    """Load a customer/sales or store file as a plain DataFrame with latitude/longitude columns.

    Point geometries (GeoJSON/Shapefile/GeoParquet) are converted to
    latitude/longitude in EPSG:4326. Files that already have the columns are
    returned untouched; validation happens later in :mod:`src.validation`.
    """
    obj = read_any(source, filename)
    if hasattr(obj, "geometry") and getattr(obj, "_geometry_column_name", None) in getattr(obj, "columns", []):
        gdf = obj
        has_cols = {"latitude", "longitude"} <= {str(c).lower() for c in gdf.columns}
        if not has_cols:
            if gdf.crs is None:
                raise ValidationError("The file has point geometries but no CRS; cannot derive latitude/longitude.")
            if not gdf.geom_type.isin(["Point"]).all():
                raise ValidationError("Customer/store files must contain Point geometries (or latitude/longitude columns).")
            g4326 = gdf.to_crs("EPSG:4326")
            out = pd.DataFrame(gdf.drop(columns="geometry"))
            out["longitude"] = g4326.geometry.x.to_numpy()
            out["latitude"] = g4326.geometry.y.to_numpy()
            return out
        return pd.DataFrame(gdf.drop(columns="geometry"))
    return obj


def load_regions(source, filename: str | None = None):
    """Load and validate administrative boundaries (EPSG:4326 GeoDataFrame with 'region' + geometry)."""
    obj = read_any(source, filename)
    if not hasattr(obj, "geometry"):
        raise ValidationError("Boundary files must be GeoJSON, a zipped Shapefile or GeoParquet (they need polygon geometry).")
    return validate_regions_gdf(obj)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
EXPORT_FORMATS = {
    "CSV": ("csv", "text/csv"),
    "Excel (.xlsx)": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "GeoJSON": ("geojson", "application/geo+json"),
    "Shapefile (.zip)": ("zip", "application/zip"),
    "GeoParquet": ("parquet", "application/octet-stream"),
}


def _stringify_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime"):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return out


def export_table(df: pd.DataFrame, fmt: str, basename: str = "export") -> tuple[bytes, str, str]:
    """Serialise ``df`` to bytes. Returns ``(data, filename, mime_type)``.

    Geo formats need latitude/longitude columns (points) or a geometry column.
    Shapefile column names are truncated to 10 characters by the format itself.
    """
    if fmt not in EXPORT_FORMATS:
        raise ValidationError(f"Unknown export format '{fmt}'.")
    ext, mime = EXPORT_FORMATS[fmt]
    fname = f"{basename}.{ext}"
    plain = pd.DataFrame(df.drop(columns="geometry")) if "geometry" in df.columns else df

    if ext == "csv":
        return plain.to_csv(index=False).encode("utf-8"), fname, mime
    if ext == "xlsx":
        buf = io.BytesIO()
        _stringify_datetimes(plain).to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue(), fname, mime

    gpd = _gpd()
    if hasattr(df, "geometry") and "geometry" in df.columns:
        gdf = df
    elif {"latitude", "longitude"} <= set(df.columns):
        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    else:
        raise ValidationError("This table has no geometry or latitude/longitude columns, so it cannot be exported as a geo format.")
    gdf = _stringify_datetimes(gdf)
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    if ext == "geojson":
        return gdf.to_json().encode("utf-8"), fname, mime
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        if ext == "parquet":
            try:
                gdf.to_parquet(tmpdir / "out.parquet")
            except ImportError as exc:
                raise MissingDependencyError("pyarrow is required for GeoParquet export.") from exc
            return (tmpdir / "out.parquet").read_bytes(), fname, mime
        # zipped shapefile
        shp_dir = tmpdir / basename
        shp_dir.mkdir()
        gdf.to_file(shp_dir / f"{basename}.shp", driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in shp_dir.iterdir():
                zf.write(f, f.name)
        return buf.getvalue(), fname, mime
