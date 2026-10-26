# Geospatial Business Expansion & Location Intelligence Platform

A Streamlit app that analyses geographic sales data, finds regions with **high demand but low store presence**,
ranks them with a transparent score, and suggests candidate branch locations.

> The bundled data is **synthetic**. Scores and candidate sites are **decision-support estimates**, not forecasts
> or guaranteed business outcomes.

## Features

| Area | What it does |
|---|---|
| Dashboard | KPI cards, date / region / category / order-value filters, downloadable summary |
| Maps | Choropleth (switchable metric), customers, own stores, competitors, candidate sites, sales-weighted heatmap, store coverage, DBSCAN hotspots, layer control + legend |
| Opportunity engine | 4-component score (demand, store gap, distance to nearest store, competition) with **adjustable weights** and a plain-language explanation per region |
| Spatial analysis | Spatial join & point-in-polygon, polygon intersection (`overlay`), nearest-neighbour (KD-tree), buffers, distance-based coverage, underserved areas, DBSCAN |
| Formats | Read CSV, XLSX, GeoJSON, zipped Shapefile, GeoParquet. Export CSV, XLSX, GeoJSON, Shapefile (zip), GeoParquet |
| Raster (optional) | GeoTIFF / COG: windowed + decimated reads, heuristic COG check, map overlay, value sampling at candidate sites |
| Validation | Missing columns, bad/missing coordinates, empty files, unsupported types -> clear messages. Nothing is silently invented |

## Architecture

```
            +-------------------+
 CSV/XLSX/  |  data_loader.py  |  read_any / load_points_table / load_regions / export_table
 GeoJSON/   +---------+---------+
 SHP/Parquet          v
            +-------------------+
            |  validation.py   |  schema + coordinate + geometry checks -> cleaned data + warnings
            +---------+---------+
                      v
            +-------------------+     +--------------------+
            |   pipeline.py    |---->| spatial_analysis.py|  joins, UTM projection, KD-tree, buffers, DBSCAN
            | prepare -> analyze|     +--------------------+
            |                   |     +--------------------+
            |                   |---->|opportunity_scoring |  features, normalisation, weights, explanations
            +---------+---------+     +--------------------+
                      v
        +-------------+--------------+
        v                            v
  mapping.py (Folium)        reporting.py (Plotly, CSV, summary)        raster_utils.py (optional COG)
        \                            /
                  app.py (Streamlit UI)
```

* **Stage 1 `prepare_data`** (cached per upload): validation, region spatial join, nearest-store distances.
* **Stage 2 `analyze`** (re-run when filters/settings change): filtering, coverage, scoring, hotspots, candidates.
* **CRS rule:** inputs are WGS84 lon/lat; every distance, buffer and area is computed after projecting to the
  UTM zone of the data's centre (metres). Maps use lon/lat again.

## Scoring method

`score = 100 x sum(weight_i x c_i)`, each `c_i` min-max normalised to 0-1 across the regions in view:

| Component | Raw measure | High value means |
|---|---|---|
| demand | sum of `demand` (or `sales` as proxy) | more demand |
| store_gap | 1 - norm(own stores / demand) | few stores for the demand |
| distance | mean customer distance to nearest own store | customers far from stores |
| competition | 1 - norm(competitor count) | few competitors |

Components with no input data (no stores, no competitor rows) are **excluded and the remaining weights
re-normalised**. A region is flagged *high demand / low presence* when both its demand and store-gap
components are >= 0.5. Scores are **relative** to the other regions displayed.

**Limitations:** straight-line (not travel-time) distances, no rent / zoning / cannibalisation / seasonality
modelling, and results are only as good as the input data.

## Windows setup (PowerShell, VS Code)

Requires Python 3.11 or newer (`python --version`).

```powershell
# 1. Create the project folder (or unzip the provided archive here)
mkdir C:\Projects\geospatial-business-expansion
cd C:\Projects\geospatial-business-expansion

# 2. Create and activate a virtual environment
python -m venv .venv
# If activation is blocked, run once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the app  (opens http://localhost:8501)
streamlit run app.py

# 5. Run the tests
pytest -v

# (optional) regenerate the synthetic sample data
python scripts\generate_sample_data.py
```

In VS Code: *File -> Open Folder*, press `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose `.venv`,
then use the integrated terminal (*View -> Terminal*) for the commands above.

## Verifying each feature

1. **Overview** - KPIs show revenue, orders, customers, stores and opportunity regions; changing the sidebar
   date or category filter changes them.
2. **Map** - toggle layers via the control (top right). With sample data, zones Z03, Z08, Z14 and Z19 should glow
   on the heatmap and sit outside the blue coverage circles.
3. **Opportunities** - those four zones rank at the top; move the weight sliders and watch ranks change. Download the CSV.
4. **Spatial analysis** - each tab maps to one technique (join, nearest neighbour, buffers, intersection, DBSCAN).
5. **Data & upload** - switch the source to *Upload*, upload `data/sample_sales.csv`, and then try a broken file
   (delete the `sales` column) to see the validation message.
6. **Raster (COG)** - upload any GeoTIFF; the app reports whether it is tiled with overviews.

## Prepare for GitHub

```powershell
git init
git add .
git commit -m "Initial commit: geospatial business expansion platform"
git branch -M main
# Create an empty repository on github.com first, then:
git remote add origin https://github.com/<your-username>/geospatial-business-expansion.git
git push -u origin main
```

Tip: add screenshots to `assets/` and link them here for your portfolio.

## Deploy to Streamlit Community Cloud

1. Push the repository to GitHub (public, or private with access granted).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app**.
3. Choose the repository, branch `main`, main file `app.py`.
4. Under **Advanced settings** pick **Python 3.11** (or 3.12).
5. Click **Deploy**. The first build installs `requirements.txt` (a few minutes).

No secrets or API keys are required. Map tiles come from free public servers (CARTO / OpenStreetMap).
Community Cloud has limited memory, so keep uploads reasonably sized.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install` fails on `rasterio` | Raster support is optional: delete that line from `requirements.txt`; the app disables the Raster page |
| `.venv\Scripts\Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Map shows but is blank | Check your internet connection (tiles are fetched online) |
| "No orders match the current filters" | Widen the sidebar filters |

## Project layout

```
app.py                      Streamlit UI
src/data_loader.py          read/export CSV, XLSX, GeoJSON, Shapefile, GeoParquet
src/validation.py           schema, coordinate and geometry validation
src/spatial_analysis.py     spatial join, CRS handling, nearest neighbour, buffers, DBSCAN
src/opportunity_scoring.py  features, weights, scores, explanations, candidate sites
src/pipeline.py             prepare_data() and analyze() orchestration
src/mapping.py              Folium map builder
src/reporting.py            Plotly charts, tables, summary text
src/raster_utils.py         optional COG / raster support
scripts/generate_sample_data.py   deterministic synthetic data generator
tests/                      pytest suite
data/                       synthetic sample data (+ schema docs)
```
