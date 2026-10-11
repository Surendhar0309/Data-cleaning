"""
FastAPI application for the Universal AI-Powered Data Cleaning Platform.
"""

import io
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from backend import data_cleaner as cleaner
from backend import feature_engineering as fe
from backend.data_profiler import classify_columns, compute_missing_values, profile_dataset
from backend.dataset_loader import (
    dataframe_to_csv_bytes,
    dataframe_to_json_bytes,
    load_dataset,
)
from backend.quality_analyzer import calculate_quality_score
from backend.report_generator import generate_cleaning_report, generate_recommendations

app = FastAPI(title="Universal Data Cleaning Platform", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> {"df", "original", "metadata", "log"}
SESSIONS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------- helpers ---

def _clean_value(v: Any) -> Any:
    """Make any value JSON-safe (NaN/NaT -> None, numpy -> python)."""
    if isinstance(v, dict):
        return {str(k): _clean_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_value(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _get(session_id: str) -> Dict[str, Any]:
    s = SESSIONS.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


def _state(s: Dict[str, Any]) -> Dict[str, Any]:
    df = s["df"]
    return {
        "profile": _clean_value(profile_dataset(df)),
        "quality_score": _clean_value(calculate_quality_score(df)),
    }


def _run(s: Dict[str, Any], func, *args, **kwargs):
    """Run a cleaning op on a copy so a failure never corrupts the session."""
    try:
        df, info = func(s["df"].copy(), *args, **kwargs)
    except HTTPException:
        raise
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    s["df"] = df
    s["log"].append(info if isinstance(info, str) else str(info))
    return info


# ----------------------------------------------------------------- models ---

class FillMissingReq(BaseModel):
    session_id: str
    column: str
    method: str
    custom_value: Optional[Any] = None


class DropMissingReq(BaseModel):
    session_id: str
    columns: Optional[List[str]] = None
    axis: str = "rows"
    threshold: float = 50.0


class DuplicatesReq(BaseModel):
    session_id: str
    subset: Optional[List[str]] = None
    keep: str = "first"


class OutlierReq(BaseModel):
    session_id: str
    column: str
    method: str = "iqr"
    action: str = "detect"  # detect | remove | cap
    factor: float = 1.5
    threshold: float = 3.0


class ConvertReq(BaseModel):
    session_id: str
    column: str
    target_type: str


class EncodeReq(BaseModel):
    session_id: str
    column: str
    method: str = "label_encode"
    drop_first: bool = False


class ScaleReq(BaseModel):
    session_id: str
    column: str
    method: str = "min_max"


class DatetimeReq(BaseModel):
    session_id: str
    column: str
    features: Optional[List[str]] = None


# -------------------------------------------------------------- endpoints ---

@app.get("/")
def health():
    return {"status": "ok", "service": "Universal Data Cleaning Platform"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        try:
            df, meta = load_dataset(content, file.filename or "data.csv")
        except UnicodeDecodeError:
            df, meta = load_dataset(content, file.filename or "data.csv", encoding="latin-1")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty and len(df.columns) == 0:
        raise HTTPException(status_code=400, detail="File contains no data")

    sid = uuid.uuid4().hex[:8]
    SESSIONS[sid] = {"df": df, "original": df.copy(), "metadata": meta, "log": []}
    return {
        "session_id": sid,
        "metadata": _clean_value(meta),
        **_state(SESSIONS[sid]),
        "recommendations": _clean_value(generate_recommendations(df)),
    }


@app.get("/profile/{session_id}")
def profile(session_id: str):
    return _state(_get(session_id))


@app.get("/quality/{session_id}")
def quality(session_id: str):
    return {"quality_score": _clean_value(calculate_quality_score(_get(session_id)["df"]))}


@app.get("/recommendations/{session_id}")
def recommendations(session_id: str):
    return {"recommendations": _clean_value(generate_recommendations(_get(session_id)["df"]))}


@app.get("/missing/{session_id}")
def missing(session_id: str):
    return {"missing": _clean_value(cleaner.get_missing_summary(_get(session_id)["df"]))}


@app.post("/clean/fill-missing")
def fill_missing(req: FillMissingReq):
    s = _get(req.session_id)
    msg = _run(s, cleaner.fill_missing, req.column, req.method, req.custom_value)
    return {"message": msg, **_state(s)}


@app.post("/clean/drop-missing")
def drop_missing(req: DropMissingReq):
    s = _get(req.session_id)
    if req.axis == "columns":
        msg = _run(s, cleaner.drop_missing_columns, req.threshold)
    else:
        msg = _run(s, cleaner.drop_missing_rows, req.columns)
    return {"message": msg, **_state(s)}


@app.get("/duplicates/{session_id}")
def duplicates(session_id: str):
    return {"duplicates": cleaner.detect_duplicates(_get(session_id)["df"])}


@app.post("/clean/remove-duplicates")
def remove_duplicates(req: DuplicatesReq):
    s = _get(req.session_id)
    msg = _run(s, cleaner.remove_duplicates, req.subset, req.keep)
    return {"message": msg, **_state(s)}


@app.post("/clean/outliers")
def outliers(req: OutlierReq):
    s = _get(req.session_id)
    df = s["df"]
    kwargs = {"factor": req.factor} if req.method == "iqr" else {"threshold": req.threshold}

    if req.action == "detect":
        try:
            if req.method == "iqr":
                info = cleaner.detect_outliers_iqr(df, req.column, req.factor)
            elif req.method == "zscore":
                info = cleaner.detect_outliers_zscore(df, req.column, req.threshold)
            else:
                raise ValueError(f"Unknown method: '{req.method}'")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"outliers": _clean_value(info)}

    if req.action == "remove":
        msg = _run(s, cleaner.remove_outliers, req.column, req.method, **kwargs)
    elif req.action == "cap":
        msg = _run(s, cleaner.cap_outliers, req.column, req.method, **kwargs)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: '{req.action}'")
    return {"message": msg, **_state(s)}


@app.get("/type-issues/{session_id}")
def type_issues(session_id: str):
    return {"issues": cleaner.detect_type_issues(_get(session_id)["df"])}


@app.post("/clean/convert-type")
def convert_type(req: ConvertReq):
    s = _get(req.session_id)
    msg = _run(s, cleaner.convert_column_type, req.column, req.target_type)
    return {"message": msg, **_state(s)}


@app.post("/feature/encode")
def encode(req: EncodeReq):
    s = _get(req.session_id)
    if req.method == "label_encode":
        info = _run(s, fe.label_encode, req.column)
    elif req.method == "one_hot_encode":
        info = _run(s, fe.one_hot_encode, req.column, req.drop_first)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown encoding: '{req.method}'")
    return {"info": _clean_value(info), **_state(s)}


@app.post("/feature/scale")
def scale(req: ScaleReq):
    s = _get(req.session_id)
    funcs = {
        "min_max": fe.min_max_scale, "min_max_scale": fe.min_max_scale,
        "standard": fe.standard_scale, "standard_scale": fe.standard_scale,
        "log": fe.log_transform, "log_transform": fe.log_transform,
    }
    func = funcs.get(req.method)
    if func is None:
        raise HTTPException(status_code=400, detail=f"Unknown scaling method: '{req.method}'")
    info = _run(s, func, req.column)
    return {"info": _clean_value(info), **_state(s)}


@app.post("/feature/datetime")
def datetime_features(req: DatetimeReq):
    s = _get(req.session_id)
    info = _run(s, fe.extract_datetime_features, req.column, req.features)
    return {"info": _clean_value(info), **_state(s)}


@app.get("/feature/available/{session_id}")
def available(session_id: str):
    return {"operations": _clean_value(fe.get_available_operations(_get(session_id)["df"]))}


@app.get("/report/{session_id}")
def report(session_id: str):
    s = _get(session_id)
    rep = generate_cleaning_report(s["df"], s["original"])
    rep["operations_log"] = s["log"]
    return {"report": _clean_value(rep)}


@app.get("/data/{session_id}")
def data(session_id: str, rows: int = 50):
    df = _get(session_id)["df"]
    rows = max(1, min(rows, 1000))
    head = df.head(rows)
    records = head.astype(object).where(head.notna(), None).to_dict(orient="records")
    return {
        "columns": [str(c) for c in df.columns],
        "data": _clean_value(records),
        "total_rows": int(len(df)),
    }


@app.get("/download/{session_id}")
def download(session_id: str, format: str = "csv"):
    df = _get(session_id)["df"]
    if format == "json":
        return Response(
            dataframe_to_json_bytes(df),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="cleaned_{session_id}.json"'},
        )
    if format == "csv":
        return Response(
            dataframe_to_csv_bytes(df),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="cleaned_{session_id}.csv"'},
        )
    raise HTTPException(status_code=400, detail="format must be 'csv' or 'json'")


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _get(session_id)
    del SESSIONS[session_id]
    return {"message": "Session deleted"}
