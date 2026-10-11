# Universal AI-Powered Data Cleaning and Preprocessing Platform

> A dataset-independent, modular platform that automatically profiles, cleans, transforms, and assesses the quality of **any** tabular dataset — without hardcoded column names or assumptions.

---

## Research Paper Alignment

**Title:** *"Universal AI-Powered Data Cleaning and Preprocessing Framework for Heterogeneous Datasets"*

### Research Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | Automatic Dataset Profiling | Dynamically detects column types, distributions, correlations, and missing patterns |
| 2 | Adaptive Cleaning Pipeline | Applies context-aware cleaning based on detected data characteristics |
| 3 | Dynamic Feature Engineering | Encoding, scaling, and transformations without fixed column dependencies |
| 4 | Data Quality Assessment | Weighted scoring (0–100) across completeness, uniqueness, outliers, and consistency |
| 5 | Dataset-Independent Architecture | Modular backend with zero hardcoded assumptions about input data |

### Abstract

> Data preprocessing remains one of the most time-consuming phases in any data science workflow, often requiring manual inspection and domain-specific scripting. We present a universal, AI-powered framework that automatically profiles heterogeneous tabular datasets, identifies quality issues, recommends cleaning strategies, and applies adaptive preprocessing pipelines — all without requiring predefined schemas or column-specific configurations. The system employs a modular architecture comprising dataset loading, automatic profiling, intelligent cleaning, dynamic feature engineering, quality scoring, and AI-driven recommendations. Evaluation across diverse datasets demonstrates the framework's ability to significantly reduce preprocessing time while maintaining or improving data quality scores compared to manual approaches.

---

## Features

### Core Capabilities

- **Multi-format Support** — CSV, Excel (.xlsx), TSV, JSON
- **Automatic Dataset Analysis** — Rows, columns, types, distributions, correlations
- **Universal Data Cleaning** — Missing values, duplicates, outliers, type corrections
- **Feature Engineering** — Label/One-Hot encoding, Min-Max/Standard scaling, Log transform, Datetime extraction
- **Data Quality Score** — 0–100 weighted metric (completeness, uniqueness, outliers, consistency)
- **AI Recommendations** — Actionable suggestions prioritized by severity
- **Modern Dashboard** — Real-time visualizations, heatmaps, correlation matrices
- **Download Cleaned Data** — Export as CSV or JSON

### Data Cleaning Methods

| Category | Operations |
|---|---|
| Missing Values | Mean, Median, Mode, Forward Fill, Backward Fill, Drop rows/columns |
| Duplicates | Detect & remove (configurable keep strategy) |
| Outliers | IQR method, Z-Score method, Remove or Cap (winsorize) |
| Type Correction | Auto-detect issues, convert to numeric/datetime/category |

### Feature Engineering

| Operation | Description |
|---|---|
| Label Encoding | Map categories to integers |
| One-Hot Encoding | Binary columns per category (with cardinality limit) |
| Min-Max Scaling | Normalize to [0, 1] |
| Standard Scaling | Z-score normalization (mean=0, std=1) |
| Log Transformation | Log1p with automatic shift for negative values |
| Datetime Extraction | Year, month, day, hour, dayofweek, quarter, is_weekend |

---

## Project Structure

```
Data-Cleaning_env/
├── backend/
│   ├── __init__.py              # Package init
│   ├── app.py                   # FastAPI main application
│   ├── dataset_loader.py        # Multi-format file loading
│   ├── data_profiler.py         # Automatic dataset analysis
│   ├── data_cleaner.py          # Universal cleaning operations
│   ├── feature_engineering.py   # Encoding, scaling, transforms
│   ├── quality_analyzer.py      # Data quality scoring (0-100)
│   └── report_generator.py      # AI recommendations & reports
├── uploads/                     # Temporary upload directory
├── index.html                   # Frontend dashboard
├── app.js                       # Frontend logic
├── style.css                    # Dashboard styles
├── server.py                    # Server entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container deployment
└── README.md                    # This file
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Surendhar0309/Data-Cleaning_env.git
cd Data-Cleaning_env

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
numpy==1.26.2
pandas==2.1.4
openpyxl==3.1.2
python-multipart==0.0.6
scikit-learn==1.3.2
```

---

## Running Locally

### 1. Start the Backend Server

```bash
python server.py
```

Server starts at `http://localhost:7860`. Interactive API docs at `http://localhost:7860/docs`.

### 2. Open the Frontend

Open `index.html` in your browser, or serve with:

```bash
python -m http.server 8080
# Then open http://localhost:8080
```

### 3. Upload a Dataset

1. Click "Browse Files" or drag & drop any CSV/Excel/TSV/JSON file
2. The platform automatically profiles your data and displays the dashboard
3. Use the tabs to clean, transform, and assess your dataset
4. Download the cleaned result

---

## API Documentation

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload dataset (multipart form) |
| GET | `/profile/{session_id}` | Full dataset profile |
| GET | `/quality/{session_id}` | Data quality score |
| GET | `/recommendations/{session_id}` | AI recommendations |
| GET | `/missing/{session_id}` | Missing value summary |
| POST | `/clean/fill-missing` | Fill missing values |
| POST | `/clean/drop-missing` | Drop rows/columns with missing |
| GET | `/duplicates/{session_id}` | Detect duplicates |
| POST | `/clean/remove-duplicates` | Remove duplicates |
| POST | `/clean/outliers` | Detect/remove/cap outliers |
| GET | `/type-issues/{session_id}` | Data type issues |
| POST | `/clean/convert-type` | Convert column type |
| POST | `/feature/encode` | Apply encoding |
| POST | `/feature/scale` | Apply scaling |
| POST | `/feature/datetime` | Extract datetime features |
| GET | `/feature/available/{session_id}` | Available operations |
| GET | `/report/{session_id}` | Cleaning report |
| GET | `/download/{session_id}` | Download cleaned data |
| GET | `/data/{session_id}` | Data preview |
| DELETE | `/session/{session_id}` | Delete session |

### Example: Upload and Clean

```python
import requests

# Upload
with open("my_data.csv", "rb") as f:
    resp = requests.post("http://localhost:7860/upload", files={"file": f})
    session_id = resp.json()["session_id"]

# Fill missing values
requests.post("http://localhost:7860/clean/fill-missing", json={
    "session_id": session_id,
    "column": "age",
    "method": "median"
})

# Get quality score
quality = requests.get(f"http://localhost:7860/quality/{session_id}").json()
print(f"Quality Score: {quality['quality_score']['overall_score']}/100")

# Download cleaned data
resp = requests.get(f"http://localhost:7860/download/{session_id}")
with open("cleaned_data.csv", "wb") as f:
    f.write(resp.content)
```

---

## Docker Deployment

```bash
docker build -t data-cleaning-platform .
docker run -p 7860:7860 data-cleaning-platform
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                          │
│  (HTML/CSS/JS — Upload, Visualize, Interact, Download)        │
└────────────────────────────┬─────────────────────────────────┘
                             │ REST API
┌────────────────────────────▼─────────────────────────────────┐
│                    FastAPI Application                         │
│                    (backend/app.py)                            │
├──────────┬──────────┬──────────┬──────────┬──────────────────┤
│ Dataset  │  Data    │  Data    │ Feature  │ Quality  Report  │
│ Loader   │ Profiler │ Cleaner  │ Engineer │ Analyzer Generator│
└──────────┴──────────┴──────────┴──────────┴──────────────────┘
```

---

## License

MIT
