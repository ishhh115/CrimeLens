# CrimeLens — Crime Pattern Analysis & Risk Prediction for Indian States

> An applied ML system for state-level crime analysis, geospatial visualization, and risk scoring using 14 years of NCRB/IPC data across 35 Indian states and union territories.

---

## Overview

CrimeLens is a full-stack data analysis platform that transforms raw, multi-source NCRB (National Crime Records Bureau) CSV data into actionable insights through machine learning, statistical anomaly detection, and interactive geospatial dashboards.

The project addresses a real challenge in public policy research: NCRB datasets are released annually in inconsistent formats with missing values, misaligned state codes, and no standardized schema across years. CrimeLens builds a reproducible pipeline from raw ingestion to trained model — entirely from scratch, without a pre-cleaned dataset.

---

## Research Motivation

Crime data in India is published at the state level but rarely analyzed in a way that accounts for temporal trends, inter-state variance, or crime composition simultaneously. Existing tools either visualize raw counts (which conflate population differences) or require manual preprocessing per year.

This project explores:
- Can engineered features (solve rate, violent-crime ratio, YoY delta) produce meaningful state-level risk stratification?
- How do anomaly detection signals compare to raw crime counts as policy indicators?
- What crime types drive overall IPC totals, and how has the composition shifted over 14 years?

---

## Dataset

| Source | Coverage | Records |
|--------|----------|---------|
| NCRB IPC Annual Reports | 2001 – 2014 | 35 states & UTs |
| IPC-specific CSVs (2013–2014) | Crime-type breakdown | 40+ IPC categories |
| India States GeoJSON | Administrative boundaries | Polygon data for choropleth |

**Preprocessing challenges handled:**
- Inconsistent state name formatting across years (`DELHI` vs `Delhi (UT)`)
- Missing values in low-population UTs for specific crime categories
- Multi-year aggregation without double-counting reorganized states
- Normalization across states with 10x+ population variance

---

## ML Pipeline

### Feature Engineering (`model.py`)

Raw crime counts are transformed into the following features before model training:

| Feature | Description |
|---------|-------------|
| `violent_crime_ratio` | Murders + Dacoity + Robbery as % of total IPC |
| `solve_rate` | Chargesheeted cases / Cases reported |
| `yoy_delta` | Year-over-year % change in total IPC crimes |
| `women_crime_ratio` | Crimes against women as % of total |
| `crime_per_capita` | Total IPC / state population (where available) |
| `anomaly_score` | Z-score deviation from state's own historical mean |

### Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Task:** State-level risk tier classification (Low / Medium / High)
- **Training:** Aggregated historical features per state across all available years
- **Evaluation:** Accuracy accessible via `/api/accuracy` endpoint
- **Inference:** Per-state risk score normalized to [0, 1] via `/api/risk-scores`

The model is trained at server startup and cached in memory. For large datasets, consider precomputing and persisting the trained model via `joblib`.

### Anomaly Detection

Z-score based anomaly detection flags states where a given year's crime count deviates significantly from that state's own historical distribution — a more meaningful signal than raw rankings, since it captures sudden spikes rather than chronically high baselines.

---

## System Architecture

```
Raw CSVs (NCRB/IPC)
        │
        ▼
  analysis.py ──── Data loading, cleaning, aggregation, chart generation
        │
        ├──► Folium choropleth (GeoJSON overlay, risk-score fill)
        │
  model.py ──────── Feature engineering → RandomForest training → prediction helpers
        │
        ▼
    app.py (Flask) ── REST API + Jinja2 template rendering
        │
        ▼
  Frontend (HTML/CSS/JS) ── Interactive dashboards, charts, map embed
```

---

## Key Findings

- **Solve rate is the strongest differentiator** between high and low risk states — states with low solve rates consistently score higher on the risk model regardless of absolute crime volume
- **Anomaly detection outperforms raw ranking** for identifying policy-relevant spikes; several states with average overall crime counts show sharp single-year anomalies tied to specific IPC categories
- **Crimes against women trend** shows consistent YoY increase post-2012 across most states, with composition shifting toward recorded domestic violence cases

---

## Repository Layout

```
├── app.py           # Flask application and routes
├── analysis.py      # Data loading, aggregation, charts, choropleth generation
├── model.py         # Feature engineering, model training, prediction helpers
├── data/
│   ├── crime_data.csv
│   ├── ipc_2013.csv
│   ├── ipc_2014.csv
│   └── india_states.geojson
├── templates/       # Jinja2 HTML pages
└── static/          # CSS, JS, assets
```

---

## Setup & Run

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/ishhh115/CrimeLens.git
cd CrimeLens
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1
# or cmd.exe
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install flask pandas numpy scikit-learn folium
```

Or with a requirements file:
```bash
pip install -r requirements.txt
```

### Data

Place the following files in `data/` before running:

- `data/crime_data.csv` — main NCRB dataset (columns: `STATE/UT`, `YEAR`, `TOTAL IPC CRIMES`, individual crime columns)
- `data/ipc_2013.csv`, `data/ipc_2014.csv` — IPC category breakdowns
- `data/india_states.geojson` — state boundary polygons

### Running

```bash
# Simple
python app.py

# Flask CLI — PowerShell
$env:FLASK_APP = "app.py"
flask run

# Flask CLI — cmd.exe
set FLASK_APP=app.py
flask run
```

App runs at `http://127.0.0.1:5000` with `debug=True` by default (see `app.py`).

---

## Pages & API Reference

### Frontend Pages

| Route | Description |
|-------|-------------|
| `/` | Home page |
| `/about` | About page |
| `/insights` | Insights dashboard |
| `/map` | Interactive choropleth map |

### JSON API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/state-crimes` | Total IPC crimes aggregated per state |
| `GET /api/year-trend` | Nationwide yearly crime totals (2001–2014) |
| `GET /api/crime-types` | IPC category breakdown |
| `GET /api/women-crimes` | Crimes against women — 14-year trend |
| `GET /api/risk-scores` | Normalized ML risk score per state |
| `GET /api/anomalies` | Z-score anomaly flags per state-year |
| `GET /api/predict/<state>` | Risk tier prediction for a given state (uppercase state name) |
| `GET /api/policy-insights` | Suggested policy focus area per state |
| `GET /api/action-brief/<state>` | Intervention summary for a state |
| `GET /api/accuracy` | Model accuracy on held-out evaluation set |
| `GET /api/choropleth` | Rendered Folium choropleth HTML |
| `GET /api/states` | List of all states in the dataset |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, Pandas, NumPy |
| Machine learning | Scikit-learn (Random Forest) |
| Geospatial viz | Folium, GeoJSON |
| Statistical analysis | NumPy (z-score), custom aggregation |
| Backend | Flask, REST API |
| Frontend | HTML5, CSS3, JavaScript, Chart.js |

---

## Limitations & Future Work

- Dataset ends at 2014 due to NCRB public release availability; extending to recent years would significantly improve model relevance
- Population normalization is approximate; district-level granularity would improve per-capita features
- The Random Forest model is trained on aggregate features — a time-series approach (LSTM or ARIMA) could better capture temporal dependencies
- No causal inference; correlations between features and risk tiers do not imply causation
- For production use, serialize the trained model with `joblib` rather than retraining on every server start

---

## Notes

- Choropleth generation uses Folium and serves HTML from `/api/choropleth` and `/map`
- The prediction model and analysis functions are heuristics for research demonstration and should not be used as the sole basis for operational decisions
- If you change templates or static assets, restart the server (or rely on Flask debug auto-reload)

---

## License

This project is for research and educational purposes. Data sourced from publicly available NCRB annual reports. Add a `LICENSE` file and contribution guidelines if you plan to publish or share widely.
