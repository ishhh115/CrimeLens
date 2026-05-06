# CRIMELENS

CRIMELENS is a small Flask web application that analyzes Indian crime data (NCRB/IPC CSVs), provides state-level insights, an interactive choropleth map, and a simple risk-prediction model for states.

**Features**
- State-wise crime aggregation and trends
- Crime-type breakdown and crimes-against-women trend
- Choropleth map (Folium) using `data/india_states.geojson`
- Simple RandomForest-based risk prediction per state
- Several JSON API endpoints for integration or AJAX calls

**Repository layout**
- `app.py`: Flask application and routes
- `analysis.py`: data loading, aggregation, charts, choropleth generation
- `model.py`: feature engineering, model training, prediction helpers
- `data/`: input CSVs and GeoJSON
- `templates/`, `static/`: frontend assets and pages

**Prerequisites**
- Python 3.8+
- pip

**Key Python packages**
- `flask`
- `pandas`
- `numpy`
- `scikit-learn`
- `folium`

You can install them directly:

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# or cmd.exe
venv\Scripts\activate
pip install flask pandas numpy scikit-learn folium
```

(Or create a `requirements.txt` with the above packages and run `pip install -r requirements.txt`.)

**Run (development)**
From the project root (where `app.py` lives):

```bash
# Simple: run directly
python app.py

# Or use Flask CLI (PowerShell)
$env:FLASK_APP = "app.py"
flask run

# Or cmd.exe
set FLASK_APP=app.py
flask run
```

By default the app runs with `debug=True` (see `app.py`). The server will be available at `http://127.0.0.1:5000/`.

**Available pages & API endpoints**
- `/` — Home page (templates/index.html)
- `/about` — About page
- `/insights` — Insights page
- `/map` — Choropleth map view

API (JSON) endpoints (useful for AJAX or integration):
- `/api/state-crimes` — Total crimes per state
- `/api/year-trend` — Yearly totals
- `/api/crime-types` — Crime-type breakdown
- `/api/women-crimes` — Crimes against women trend
- `/api/predict/<state>` — Risk prediction for a state (use uppercase state code/name)
- `/api/states` — List of all states in dataset
- `/api/risk-scores` — Normalized risk scores per state
- `/api/anomalies` — Detected anomalies (z-score based)
- `/api/policy-insights` — Suggested policy focus per state
- `/api/accuracy` — Model accuracy estimate
- `/api/choropleth` — Returns rendered HTML for Folium choropleth
- `/api/action-brief/<state>` — Action brief & interventions for a state

**Data**
Place the provided CSVs and GeoJSON in the `data/` folder:
- `data/crime_data.csv`
- `data/ipc_2013.csv`
- `data/ipc_2014.csv`
- `data/india_states.geojson`

The app expects columns such as `STATE/UT`, `YEAR`, `TOTAL IPC CRIMES`, and individual crime columns used in `analysis.py`.

**Notes & caveats**
- Choropleth generation uses `folium` and serves HTML from `/api/choropleth` and `/map`.
- `model.py` trains a RandomForest on aggregated historical data at startup (cached after first request). For large datasets, consider precomputing or persisting the trained model.
- The prediction model and analysis functions are simple heuristics for demonstration and should not be used as the sole basis for operational decisions.

**Development tips**
- If you change templates or static assets, restart the server (or rely on Flask debug auto-reload).
- To evaluate model accuracy: visit `/api/accuracy`.

**Contributing / License**
This is an example/demo project. Add a `LICENSE` and contribution guidelines if you plan to publish or share.

---
Generated README for local development and quick reference.
