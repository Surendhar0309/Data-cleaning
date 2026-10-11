"""
Data Profiler Module
Automatic dataset analysis: column type detection, missing values,
distributions and correlations. No hardcoded column names.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _looks_like_datetime(s: pd.Series) -> bool:
    """True if >=90% of a non-numeric text sample parses as a date."""
    sample = s.dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    if pd.to_numeric(sample, errors="coerce").notna().any():
        return False
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.9


def classify_columns(df: pd.DataFrame, cat_threshold: int = 50) -> Dict[str, List[str]]:
    """Classify columns into numerical, categorical, datetime and text."""
    result: Dict[str, List[str]] = {
        "numerical": [], "categorical": [], "datetime": [], "text": [],
    }
    n = len(df)
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            result["categorical"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            result["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            result["numerical"].append(col)
        elif _looks_like_datetime(s):
            result["datetime"].append(col)
        else:
            nunique = s.nunique(dropna=True)
            # many distinct values relative to rows -> free text
            if nunique > cat_threshold and n > 0 and nunique / n > 0.5:
                result["text"].append(col)
            else:
                result["categorical"].append(col)
    return result


def compute_missing_values(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Per-column missing counts and percentages."""
    total = len(df)
    out = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        out[col] = {
            "count": count,
            "percentage": round(count / total * 100, 2) if total > 0 else 0.0,
        }
    return out


def compute_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Pearson correlation matrix over numerical columns (NaN -> 0)."""
    num_cols = classify_columns(df)["numerical"]
    num_cols = [c for c in num_cols if not pd.api.types.is_bool_dtype(df[c])]
    if len(num_cols) < 2:
        return {}
    corr = df[num_cols].corr().fillna(0.0)
    return {
        r: {c: round(float(corr.loc[r, c]), 4) for c in num_cols} for r in num_cols
    }


def compute_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Descriptive statistics for numerical columns."""
    stats = {}
    for col in classify_columns(df)["numerical"]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return stats


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Full dataset profile (shape matches what the frontend expects)."""
    return {
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "column_types": classify_columns(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": compute_missing_values(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "correlations": compute_correlations(df),
        "statistics": compute_statistics(df),
    }
