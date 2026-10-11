"""
Universal AI-Powered Data Cleaning and Preprocessing Platform
Main FastAPI Application

Handles any tabular dataset (CSV, Excel, TSV, JSON) with automatic
profiling, cleaning, feature engineering, and quality assessment.
"""

import io
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import pandas as pd

from backend.dataset_loader import (
    dataframe_to_csv_bytes,
    dataframe_to_json_bytes,
    load_dataset,
)
from backend.data_profiler import profile_dataset
from backend.data_cleaner import (
    cap_outliers,
    convert_column_type,
    detect_duplicates,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_type_issues,
    drop_missing_columns,
    drop_missing_rows,
    fill_missing,
    get_missing_summary,
    remove_duplicates,
    remove_outliers,
)
from backend.feature_engineering import (
    extract_datetime_features,
    get_available_operations,
    label_encode,
    log_transform,
    min_max_scale,
    one_hot_encode,
    standard_scale,
)
from backend.quality_analyzer import calculate_quality_score
from backend.report_generator import generate_cleaning_report, generate_recommendations

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Universal AI-Powered Data Cleaning Platform",
    description=(
        "A dataset-independent platform for automatic profiling, cleaning, "
        "feature engineering, and quality assessment of any tabular dataset."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session storage ─────────────────────────────────────────────────
sessions: Dict[str, Dict[str, Any]] = {}


# ── Request Models ────────────────────────────────────────────────────────────
class FillMissingRequest(BaseModel):
    session_id: str
    column: str
    method: str = Field(..., description="mean, median, mode, ffill, bfill, zero, custom")
    custom_value: Optional[Any] = None


class DropMissingRequest(BaseModel):
    session_id: str
    columns: Optional[List[str]] = None
    axis: str = Field("rows", description="'rows' or 'columns'")
    threshold: Optional[float] = Field(50.0, description="For column dropping: max missing %")


class RemoveDuplicatesRequest(BaseModel):
    session_id: str
    subset: Optional[List[str]] = None
    keep: str = "first"


class OutlierRequest(BaseModel):
    session_id: str
    column: str
    method: str = Field("iqr", description="iqr or zscore")
    action: str = Field("detect", description="detect, remove, or cap")
    factor: Optional[float] = 1.5
    threshold: Optional[float] = 3.0


class ConvertTypeRequest(BaseModel):
    session_id: str
    column: str
    target_type: str = Field(..., description="numeric, datetime, string, category, integer, float")


class EncodingRequest(BaseModel):
    session_id: str
    column: str
    method: str = Field(..., description="label_encode, one_hot_encode")
    drop_first: bool = False


class ScalingRequest(BaseModel):
    session_id: str
    column: str
    method: str = Field(..., description="min_max, standard, log")


class DatetimeRequest(BaseModel):
    session_id: str
    column: str
    features: Optional[List[str]] = None


class SessionRequest(BaseModel):
    session_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Upload a dataset first.")
    return sessions[session_id]


def _get_df(session_id: str) -> pd.DataFrame:
    return _get_session(session_id)["df"]


def _set_df(session_id: str, df: pd.DataFrame):
    sessions[session_id]["df"] = df


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "service": "Universal AI-Powered Data Cleaning Platform",
        "version": "2.0.0",
        "active_sessions": len(sessions),
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset (CSV, Excel, TSV, or JSON).
    Returns a session_id and initial profile.
    """
    filename = file.filename or "unknown.csv"
    content = await file.read()

    try:
        df, metadata = load_dataset(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = str(uuid.uuid4())[:8]
    profile = profile_dataset(df)
    quality = calculate_quality_score(df)
    recommendations = generate_recommendations(df)

    sessions[session_id] = {
        "df": df,
        "original_df": df.copy(),
        "metadata": metadata,
        "actions_log": [],
    }

    logger.info(
        f"[UPLOAD] session={session_id} file={filename} "
        f"rows={len(df)} cols={len(df.columns)} quality={quality['overall_score']}"
    )

    return {
        "status": "success",
        "session_id": session_id,
        "metadata": metadata,
        "profile": profile,
        "quality_score": quality,
        "recommendations": recommendations,
    }


@app.get("/profile/{session_id}")
async def get_profile(session_id: str):
    """Get full dataset profile for a session."""
    df = _get_df(session_id)
    profile = profile_dataset(df)
    quality = calculate_quality_score(df)
    return {"status": "success", "profile": profile, "quality_score": quality}


@app.get("/quality/{session_id}")
async def get_quality(session_id: str):
    """Get data quality score."""
    df = _get_df(session_id)
    quality = calculate_quality_score(df)
    return {"status": "success", "quality_score": quality}


@app.get("/recommendations/{session_id}")
async def get_recommendations(session_id: str):
    """Get AI-powered cleaning recommendations."""
    df = _get_df(session_id)
    recs = generate_recommendations(df)
    return {"status": "success", "recommendations": recs}


@app.get("/missing/{session_id}")
async def get_missing(session_id: str):
    """Get missing value summary."""
    df = _get_df(session_id)
    summary = get_missing_summary(df)
    return {"status": "success", "missing_values": summary}


@app.post("/clean/fill-missing")
async def api_fill_missing(request: FillMissingRequest):
    """Fill missing values in a column."""
    df = _get_df(request.session_id)
    try:
        df, msg = fill_missing(df, request.column, request.method, request.custom_value)
        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "fill_missing", "detail": msg})
        quality = calculate_quality_score(df)
        return {"status": "success", "message": msg, "quality_score": quality}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clean/drop-missing")
async def api_drop_missing(request: DropMissingRequest):
    """Drop rows or columns with missing values."""
    df = _get_df(request.session_id)
    try:
        if request.axis == "columns":
            df, msg = drop_missing_columns(df, threshold=request.threshold or 50.0)
        else:
            df, msg = drop_missing_rows(df, columns=request.columns)
        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "drop_missing", "detail": msg})
        quality = calculate_quality_score(df)
        return {"status": "success", "message": msg, "quality_score": quality, "shape": {"rows": len(df), "columns": len(df.columns)}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/duplicates/{session_id}")
async def get_duplicates(session_id: str):
    """Detect duplicates."""
    df = _get_df(session_id)
    info = detect_duplicates(df)
    return {"status": "success", "duplicates": info}


@app.post("/clean/remove-duplicates")
async def api_remove_duplicates(request: RemoveDuplicatesRequest):
    """Remove duplicate rows."""
    df = _get_df(request.session_id)
    df, msg = remove_duplicates(df, subset=request.subset, keep=request.keep)
    _set_df(request.session_id, df)
    sessions[request.session_id]["actions_log"].append({"action": "remove_duplicates", "detail": msg})
    quality = calculate_quality_score(df)
    return {"status": "success", "message": msg, "quality_score": quality, "shape": {"rows": len(df), "columns": len(df.columns)}}


@app.post("/clean/outliers")
async def api_handle_outliers(request: OutlierRequest):
    """Detect, remove, or cap outliers."""
    df = _get_df(request.session_id)
    try:
        if request.action == "detect":
            if request.method == "iqr":
                info = detect_outliers_iqr(df, request.column, factor=request.factor or 1.5)
            else:
                info = detect_outliers_zscore(df, request.column, threshold=request.threshold or 3.0)
            return {"status": "success", "outliers": info}
        elif request.action == "remove":
            kwargs = {"factor": request.factor} if request.method == "iqr" else {"threshold": request.threshold}
            df, msg = remove_outliers(df, request.column, method=request.method, **kwargs)
        elif request.action == "cap":
            kwargs = {"factor": request.factor} if request.method == "iqr" else {"threshold": request.threshold}
            df, msg = cap_outliers(df, request.column, method=request.method, **kwargs)
        else:
            raise ValueError(f"Unknown action: '{request.action}'")

        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "outliers", "detail": msg})
        quality = calculate_quality_score(df)
        return {"status": "success", "message": msg, "quality_score": quality, "shape": {"rows": len(df), "columns": len(df.columns)}}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/type-issues/{session_id}")
async def get_type_issues(session_id: str):
    """Detect data type issues."""
    df = _get_df(session_id)
    issues = detect_type_issues(df)
    return {"status": "success", "type_issues": issues}


@app.post("/clean/convert-type")
async def api_convert_type(request: ConvertTypeRequest):
    """Convert a column's data type."""
    df = _get_df(request.session_id)
    try:
        df, msg = convert_column_type(df, request.column, request.target_type)
        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "convert_type", "detail": msg})
        return {"status": "success", "message": msg}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/feature/encode")
async def api_encode(request: EncodingRequest):
    """Apply encoding to a categorical column."""
    df = _get_df(request.session_id)
    try:
        if request.method == "label_encode":
            df, info = label_encode(df, request.column)
        elif request.method == "one_hot_encode":
            df, info = one_hot_encode(df, request.column, drop_first=request.drop_first)
        else:
            raise ValueError(f"Unknown method: '{request.method}'")
        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "encode", "detail": info})
        return {"status": "success", "info": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/feature/scale")
async def api_scale(request: ScalingRequest):
    """Apply scaling to a numerical column."""
    df = _get_df(request.session_id)
    try:
        if request.method == "min_max":
            df, info = min_max_scale(df, request.column)
        elif request.method == "standard":
            df, info = standard_scale(df, request.column)
        elif request.method == "log":
            df, info = log_transform(df, request.column)
        else:
            raise ValueError(f"Unknown method: '{request.method}'")
        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "scale", "detail": info})
        return {"status": "success", "info": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/feature/datetime")
async def api_datetime(request: DatetimeRequest):
    """Extract datetime features from a column."""
    df = _get_df(request.session_id)
    try:
        df, info = extract_datetime_features(df, request.column, features=request.features)
        _set_df(request.session_id, df)
        sessions[request.session_id]["actions_log"].append({"action": "datetime_extract", "detail": info})
        return {"status": "success", "info": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/feature/available/{session_id}")
async def api_available_features(session_id: str):
    """Get available feature engineering operations."""
    df = _get_df(session_id)
    ops = get_available_operations(df)
    return {"status": "success", "operations": ops}


@app.get("/report/{session_id}")
async def get_report(session_id: str):
    """Generate cleaning report comparing original and current data."""
    session = _get_session(session_id)
    df = session["df"]
    original_df = session["original_df"]
    report = generate_cleaning_report(df, original_df)
    report["actions_log"] = session["actions_log"]
    return {"status": "success", "report": report}


@app.get("/download/{session_id}")
async def download_cleaned(session_id: str, format: str = "csv"):
    """Download the cleaned dataset."""
    df = _get_df(session_id)
    session = _get_session(session_id)
    filename = session["metadata"].get("filename", "cleaned_data")
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    if format == "json":
        content = dataframe_to_json_bytes(df)
        media_type = "application/json"
        ext = "json"
    else:
        content = dataframe_to_csv_bytes(df)
        media_type = "text/csv"
        ext = "csv"

    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={base_name}_cleaned.{ext}"},
    )


@app.get("/data/{session_id}")
async def get_data_preview(session_id: str, rows: int = 50):
    """Get a preview of the current dataset."""
    df = _get_df(session_id)
    preview = df.head(rows).fillna("null").to_dict(orient="records")
    return {
        "status": "success",
        "data": preview,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "success", "message": f"Session '{session_id}' deleted"}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
