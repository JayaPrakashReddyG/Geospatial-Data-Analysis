"""Optional raster support (GeoTIFF / Cloud Optimized GeoTIFF) via rasterio.

What "cloud optimized" means here
---------------------------------
A COG is an ordinary GeoTIFF whose internal layout (tiling + overviews + header
first) lets a client fetch only the bytes it needs - via HTTP range requests
for remote files, or windowed reads for local ones. This module:

* reads only a decimated / windowed part of the raster (never the whole array
  at full resolution), which GDAL serves from overviews when they exist;
* reports whether the file *looks* like a COG using a HEURISTIC (tiled +
  overviews). It does not run the full COG validator (use ``rio cogeo validate``
  for that) and never claims an arbitrary TIFF is cloud optimized;
* works with local files, uploaded bytes, or an http(s) URL (range requests
  happen only if the remote server supports them).

rasterio is optional: everything raises :class:`MissingDependencyError` with an
install hint instead of crashing the app.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from .validation import MissingDependencyError, ValidationError


def rasterio_available() -> bool:
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return False
    return True


def _rio():
    try:
        import rasterio
    except ImportError as exc:
        raise MissingDependencyError(
            "rasterio is not installed, so raster/COG features are disabled. Install with: pip install rasterio"
        ) from exc
    return rasterio


@contextmanager
def open_raster(source):
    """Open a raster from a path, URL, or bytes (e.g. an uploaded file)."""
    rasterio = _rio()
    if isinstance(source, (bytes, bytearray)):
        from rasterio.io import MemoryFile

        with MemoryFile(bytes(source)) as mem:
            with mem.open() as ds:
                yield ds
    else:
        try:
            with rasterio.open(str(source)) as ds:
                yield ds
        except rasterio.errors.RasterioIOError as exc:
            raise ValidationError(f"Could not open the raster: {exc}") from exc


def inspect_cog(ds) -> dict:
    """Heuristic COG structure check + basic metadata."""
    tiled = bool(ds.profile.get("tiled", False))
    overviews = ds.overviews(1) if ds.count else []
    reasons = []
    if not tiled:
        reasons.append("not internally tiled (striped layout)")
    if not overviews:
        reasons.append("no overviews (pyramids)")
    looks_cog = tiled and bool(overviews)
    return {
        "width": ds.width,
        "height": ds.height,
        "bands": ds.count,
        "dtype": ds.dtypes[0] if ds.count else None,
        "crs": ds.crs.to_string() if ds.crs else None,
        "nodata": ds.nodata,
        "compression": getattr(ds, "compression", None) and getattr(ds.compression, "value", str(ds.compression)),
        "block_shape": ds.block_shapes[0] if ds.count else None,
        "overviews": overviews,
        "looks_like_cog": looks_cog,
        "verdict": (
            f"Has COG-style structure (tiled, {len(overviews)} overview level(s)). Heuristic only - "
            "run `rio cogeo validate` for a definitive check."
            if looks_cog
            else "Not detected as a COG: " + "; ".join(reasons) + ". It can still be read, but may be slow for large/remote files."
        ),
    }


def read_preview(source, band: int = 1, max_pixels: int = 1_000_000, bbox_wgs84: tuple | None = None) -> dict:
    """Read a decimated (and optionally windowed) preview in EPSG:4326.

    ``bbox_wgs84`` = (west, south, east, north) restricts the read to a window,
    so only those blocks are fetched. Non-EPSG:4326 rasters are warped on the
    fly with a WarpedVRT (no full-size copy is made).

    Returns a dict with ``array`` (masked float array), ``bounds`` ((south, west), (north, east)
    for Folium), ``stats``, and ``info`` (see :func:`inspect_cog`).
    """
    _rio()
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, from_bounds

    with open_raster(source) as ds:
        info = inspect_cog(ds)
        if ds.crs is None:
            raise ValidationError("The raster has no coordinate reference system, so it cannot be placed on a map.")
        if not 1 <= band <= ds.count:
            raise ValidationError(f"Band {band} does not exist (raster has {ds.count} band(s)).")

        src = ds
        vrt = None
        if ds.crs.to_epsg() != 4326:
            vrt = WarpedVRT(ds, crs="EPSG:4326", resampling=Resampling.bilinear)
            src = vrt
        try:
            window = Window(0, 0, src.width, src.height)
            if bbox_wgs84 is not None:
                w, s, e, n = bbox_wgs84
                req = from_bounds(w, s, e, n, transform=src.transform)
                try:
                    window = req.intersection(window)
                except Exception as exc:  # rasterio.errors.WindowError
                    raise ValidationError("The requested area does not overlap the raster.") from exc
                window = window.round_offsets().round_lengths()
                if window.width < 1 or window.height < 1:
                    raise ValidationError("The requested area is too small or outside the raster.")

            scale = min(1.0, (max_pixels / float(window.width * window.height)) ** 0.5)
            out_w, out_h = max(1, int(window.width * scale)), max(1, int(window.height * scale))
            arr = src.read(band, window=window, out_shape=(out_h, out_w), resampling=Resampling.average, masked=True)
            left, bottom, right, top = src.window_bounds(window)
        finally:
            if vrt is not None:
                vrt.close()

    arr = np.ma.masked_invalid(arr.astype("float64"))
    valid = arr.compressed()
    stats = (
        {"min": float(valid.min()), "max": float(valid.max()), "mean": float(valid.mean()), "pixels_shown": int(arr.size)}
        if valid.size
        else {"min": None, "max": None, "mean": None, "pixels_shown": int(arr.size)}
    )
    return {"array": arr, "bounds": [[bottom, left], [top, right]], "stats": stats, "info": info}


_RAMP = np.array(
    [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=float
)  # viridis-like


def colorize(arr: np.ma.MaskedArray, opacity: float = 0.75, clip_pct: tuple = (2, 98)) -> np.ndarray:
    """Masked float array -> RGBA uint8 image (percentile-stretched viridis-like ramp)."""
    valid = arr.compressed()
    if valid.size == 0:
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    lo, hi = np.percentile(valid, clip_pct)
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((arr.filled(lo) - lo) / (hi - lo), 0, 1) * (len(_RAMP) - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(_RAMP) - 1)
    frac = (t - i0)[..., None]
    rgb = _RAMP[i0] * (1 - frac) + _RAMP[i1] * frac
    alpha = np.where(np.ma.getmaskarray(arr), 0, int(255 * opacity))
    return np.dstack([rgb, alpha]).astype(np.uint8)


def sample_at_points(source, lons, lats, band: int = 1) -> np.ndarray:
    """Raster value under each (lon, lat) point; NaN where outside the raster or nodata."""
    _rio()
    from rasterio.warp import transform

    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    with open_raster(source) as ds:
        if ds.crs is None:
            raise ValidationError("The raster has no CRS, so point sampling is not possible.")
        xs, ys = transform("EPSG:4326", ds.crs, lons.tolist(), lats.tolist())
        vals = np.full(len(lons), np.nan)
        for k, v in enumerate(ds.sample(list(zip(xs, ys)), indexes=band, masked=True)):
            if not np.ma.is_masked(v[0]):
                vals[k] = float(v[0])
    return vals
