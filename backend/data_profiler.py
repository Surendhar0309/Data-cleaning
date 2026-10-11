"""
Data Profiler Module
Automatically analyzes uploaded datasets to detect types, distributions,
and generate comprehensive summaries without any hardcoded column names.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify all columns into: numerical, categorical, datetime, text.

    Logic:
      - Datetime: columns with dtype datetime64 or that can be parsed as dates.
      - Numerical: int/float dtypes.
      - Text: object dtype with high cardinality or long strings.
      - Categorical: object dtype with low cardinality.
    """
    numerical: List[str] = []
    categorical: List[str] = []
    datetime_cols: List[str] = []
    text_cols: List[str] = []

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(dtype):
            numerical.append(col)
            continue

        # Try parsing as datetime
        if dtype == object:
            sample = df[col].dropna().head(20)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample)
                    datetime_cols.append(col)
                    continue
                except (ValueError, TypeError):
                    pass

            # Distinguish text vs categorical
            nunique = df[col].nunique()
            n_rows = len(df)
            avg_len = sample.astype(str).str.len().mean() if len(sample) > 0 else 0

            if nunique > min(50, n_rows * 0.5) or avg_len > 50:
                text_cols.append(col)
            else:
                categorical.append(col)
            continue

        # Boolean or other
        if pd.api.types.is_bool_dtype(dtype):
            categorical.append(col)
        else:
            categorical.append(col)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_cols,
        "text": text_cols,
    }


def compute_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute missing value statistics for each column."""
    total = len(df)
    missing_info = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        missing_info[col] = {
            "count": count,
            "percentage": round((count / total) * 100, 2) if total > 0 else 0.0,
        }
    return missing_info


def compute_statistics(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute descriptive statistics for numerical columns."""
    stats = {}
    for col in column_types.get("numerical", []):
        series = df[col].dropna()
        if len(series) == 0:
            stats[col] = {}
            continue
        stats[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(float(series.quantile(0.25)), 4),
            "q3": round(float(series.quantile(0.75)), 4),
            "skewness": round(float(series.skew()), 4),
        }
    return stats


def compute_correlations(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Compute correlation matrix for numerical columns."""
    num_cols = column_types.get("numerical", [])
    if len(num_cols) < 2:
        return {}
    corr_df = df[num_cols].corr()
    return {
        col: {k: round(v, 4) for k, v in row.items()}
        for col, row in corr_df.to_dict().items()
    }


def compute_unique_values(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> Dict[str, Any]:
    """Compute unique value counts for categorical columns."""
    result = {}
    for col in column_types.get("categorical", []):
        vc = df[col].value_counts(dropna=False).head(20)
        result[col] = {
            "nunique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }
    return result


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a full dataset profile.

    Returns a dict with:
      - shape (rows, cols)
      - column_types classification
      - missing values
      - statistics for numerical columns
      - correlation matrix
      - unique values for categorical columns
      - data types
    """
    column_types = classify_columns(df)
    missing = compute_missing_values(df)
    stats = compute_statistics(df, column_types)
    correlations = compute_correlations(df, column_types)
    uniques = compute_unique_values(df, column_types)

    dtypes = {col: str(df[col].dtype) for col in df.columns}

    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "column_types": column_types,
        "missing_values": missing,
        "statistics": stats,
        "correlations": correlations,
        "unique_values": uniques,
        "dtypes": dtypes,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
        "duplicate_rows": int(df.duplicated().sum()),
    }
