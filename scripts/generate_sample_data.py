"""Generate the SYNTHETIC demo datasets shipped in ``data/``.

Everything produced here is invented. The coordinates sit on an arbitrary
5 x 4 grid of "zones"; they are NOT real administrative boundaries, real
customers, or real stores. Four zones are deliberately given high demand and
(almost) no own stores so the opportunity engine has something to find.

Run from the project root:

    python scripts/generate_sample_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LON0, LAT0 = 77.40, 12.80  # south-west corner of the synthetic grid
CELL = 0.10  # cell size in degrees (~10-11 km)
NCOLS, NROWS = 5, 4
N_CUSTOMERS = 1800
N_ORDERS = 7500

# High-demand zones (multiplier on customer density).
HOT_ZONES = {"Z03": 3.0, "Z08": 2.6, "Z14": 2.8, "Z19": 2.2}
OWN_STORE_ZONES = ["Z01", "Z05", "Z06", "Z10", "Z11", "Z12", "Z16", "Z17", "Z20"]
COMPETITOR_ZONES = ["Z02", "Z06", "Z07", "Z09", "Z12", "Z14", "Z15", "Z17", "Z18", "Z19"]

# category -> (mean, sigma) of ln(order value)
CATEGORIES = {
    "Electronics": (5.2, 0.6),
    "Grocery": (3.4, 0.5),
    "Apparel": (4.2, 0.55),
    "Home": (4.6, 0.6),
    "Beauty": (3.8, 0.5),
}
CATEGORY_PROBS = [0.15, 0.35, 0.20, 0.15, 0.15]


def zone_box(index: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) of zone number ``index`` (1-based)."""
    row, col = divmod(index - 1, NCOLS)
    west = LON0 + col * CELL
    south = LAT0 + row * CELL
    return west, south, west + CELL, south + CELL


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)

    zone_ids = [f"Z{i:02d}" for i in range(1, NCOLS * NROWS + 1)]

    # ---- regions (GeoJSON) -------------------------------------------------
    features = []
    for i, zid in enumerate(zone_ids, start=1):
        w, s, e, n = zone_box(i)
        features.append(
            {
                "type": "Feature",
                "properties": {"region": zid, "region_name": f"Zone {zid[1:]}"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
                },
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    (out / "sample_regions.geojson").write_text(json.dumps(geojson), encoding="utf-8")

    # ---- customers ---------------------------------------------------------
    base_weight = rng.lognormal(mean=0.0, sigma=0.35, size=len(zone_ids))
    weight = np.array([b * HOT_ZONES.get(z, 1.0) for b, z in zip(base_weight, zone_ids)])
    per_zone = rng.multinomial(N_CUSTOMERS, weight / weight.sum())

    cust_rows = []
    cid = 1
    for i, (zid, n) in enumerate(zip(zone_ids, per_zone), start=1):
        w, s, e, nn = zone_box(i)
        fx = np.clip(0.5 + 0.22 * rng.standard_normal(n), 0.02, 0.98)
        fy = np.clip(0.5 + 0.22 * rng.standard_normal(n), 0.02, 0.98)
        for x, y in zip(fx, fy):
            cust_rows.append((f"C{cid:05d}", zid, s + y * CELL, w + x * CELL))
            cid += 1
    customers = pd.DataFrame(cust_rows, columns=["customer_id", "region", "latitude", "longitude"])

    # ---- orders ------------------------------------------------------------
    activity = rng.lognormal(0.0, 0.8, size=len(customers))
    activity /= activity.sum()
    picked = rng.choice(len(customers), size=N_ORDERS, p=activity)
    orders = customers.iloc[picked].reset_index(drop=True).copy()

    # dates in 2025 with a mild Q4 bump
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    day_w = np.where(days.month >= 10, 1.5, 1.0).astype(float)
    day_w /= day_w.sum()
    orders["date"] = rng.choice(days, size=N_ORDERS, p=day_w)
    orders["date"] = pd.to_datetime(orders["date"]).dt.strftime("%Y-%m-%d")

    cats = list(CATEGORIES)
    orders["category"] = rng.choice(cats, size=N_ORDERS, p=CATEGORY_PROBS)
    mu = orders["category"].map(lambda c: CATEGORIES[c][0]).to_numpy()
    sg = orders["category"].map(lambda c: CATEGORIES[c][1]).to_numpy()
    orders["sales"] = np.round(np.exp(rng.normal(mu, sg)), 2)
    orders["demand"] = rng.poisson(1.5, size=N_ORDERS) + 1  # units requested
    orders.insert(0, "order_id", [f"O{i:06d}" for i in range(1, N_ORDERS + 1)])
    orders = orders[
        ["order_id", "customer_id", "date", "region", "category", "sales", "demand", "latitude", "longitude"]
    ].sort_values("date")
    orders["latitude"] = orders["latitude"].round(6)
    orders["longitude"] = orders["longitude"].round(6)
    orders.to_csv(out / "sample_sales.csv", index=False)

    # ---- stores ------------------------------------------------------------
    def place(zid: str) -> tuple[float, float]:
        w, s, e, n = zone_box(int(zid[1:]))
        jitter = rng.uniform(-0.02, 0.02, size=2)
        return round(s + CELL / 2 + jitter[0], 6), round(w + CELL / 2 + jitter[1], 6)

    stores = []
    for k, zid in enumerate(OWN_STORE_ZONES, start=1):
        lat, lon = place(zid)
        stores.append((f"S{k:03d}", f"Our Store {zid}", "own", lat, lon, zid))
    for k, zid in enumerate(COMPETITOR_ZONES, start=1):
        lat, lon = place(zid)
        stores.append((f"K{k:03d}", f"Competitor {zid}", "competitor", lat, lon, zid))
    pd.DataFrame(
        stores, columns=["store_id", "store_name", "store_type", "latitude", "longitude", "region"]
    ).to_csv(out / "sample_stores.csv", index=False)

    print(f"customers={len(customers)} orders={len(orders)} stores={len(stores)} -> {out}")


if __name__ == "__main__":
    main()
