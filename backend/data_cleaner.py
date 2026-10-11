"""
Data Cleaner Module
Universal data cleaning operations that work with any tabular dataset.
No hardcoded column names — all operations are driven dynamically.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _is_text_dtype(dtype) -> bool:
    """True for object and pandas string dtypes (pandas 3 uses 'str')."""
    return pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)


# =============================================================================
# Missing Value Handling
# =============================================================================


def get_missing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get detailed missing value summary."""
    total = len(df)
    summary = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        summary[col] = {
            "missing_count": count,
            "missing_percentage": round((count / total) * 100, 2) if total > 0 else 0.0,
            "dtype": str(df[col].dtype),
        }
    return summary


def fill_missing(
    df: pd.DataFrame,
    column: str,
    method: str,
    custom_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Fill missing values in a column.

    Methods: mean, median, mode, ffill, bfill, custom, zero
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")

    original_missing = int(df[column].isnull().sum())

    if method == "mean":
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Cannot use mean fill on non-numeric column '{column}'")
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median":
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Cannot use median fill on non-numeric column '{column}'")
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        mode_val = df[column].mode()
        if len(mode_val) > 0:
            df[column] = df[column].fillna(mode_val.iloc[0])
    elif method == "ffill":
        df[column] = df[column].ffill()
    elif method == "bfill":
        df[column] = df[column].bfill()
    elif method == "zero":
        df[column] = df[column].fillna(0)
    elif method == "custom":
        if custom_value is None:
            raise ValueError("custom_value required for method='custom'")
        df[column] = df[column].fillna(custom_value)
    else:
        raise ValueError(f"Unknown method: '{method}'")

    new_missing = int(df[column].isnull().sum())
    msg = f"Filled {original_missing - new_missing} missing values in '{column}' using {method}"
    return df, msg


def drop_missing_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, str]:
    """Drop rows with missing values in specified columns (or all)."""
    original = len(df)
    if columns:
        df = df.dropna(subset=columns).reset_index(drop=True)
    else:
        df = df.dropna().reset_index(drop=True)
    dropped = original - len(df)
    return df, f"Dropped {dropped} rows with missing values"


def drop_missing_columns(df: pd.DataFrame, threshold: float = 50.0) -> Tuple[pd.DataFrame, str]:
    """Drop columns where missing percentage exceeds threshold."""
    total = len(df)
    cols_to_drop = []
    for col in df.columns:
        pct = (df[col].isnull().sum() / total) * 100
        if pct > threshold:
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop)
    return df, f"Dropped {len(cols_to_drop)} columns with >{threshold}% missing: {cols_to_drop}"


# =============================================================================
# Duplicate Handling
# =============================================================================


def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect duplicate rows."""
    dup_count = int(df.duplicated().sum())
    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": round((dup_count / len(df)) * 100, 2) if len(df) > 0 else 0.0,
        "total_rows": len(df),
    }


def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = "first"
) -> Tuple[pd.DataFrame, str]:
    """Remove duplicate rows."""
    original = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = original - len(df)
    return df, f"Removed {removed} duplicate rows (keep='{keep}')"


# =============================================================================
# Outlier Detection and Handling
# =============================================================================


def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Dict[str, Any]:
    """Detect outliers using IQR method."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    series = df[column].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    outlier_count = int(outlier_mask.sum())

    return {
        "column": column,
        "method": "IQR",
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "outlier_count": outlier_count,
        "outlier_percentage": round((outlier_count / len(df)) * 100, 2) if len(df) > 0 else 0.0,
        "outlier_indices": df.index[outlier_mask].tolist()[:100],
    }


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> Dict[str, Any]:
    """Detect outliers using Z-Score method."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    series = df[column].dropna()
    mean = float(series.mean())
    std = float(series.std())

    if std == 0:
        return {
            "column": column,
            "method": "Z-Score",
            "mean": mean,
            "std": 0.0,
            "threshold": threshold,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "outlier_indices": [],
        }

    z_scores = np.abs((df[column] - mean) / std)
    outlier_mask = z_scores > threshold
    outlier_count = int(outlier_mask.sum())

    return {
        "column": column,
        "method": "Z-Score",
        "mean": round(mean, 4),
        "std": round(std, 4),
        "threshold": threshold,
        "outlier_count": outlier_count,
        "outlier_percentage": round((outlier_count / len(df)) * 100, 2) if len(df) > 0 else 0.0,
        "outlier_indices": df.index[outlier_mask].tolist()[:100],
    }


def _outlier_mask(df: pd.DataFrame, column: str, method: str, **kwargs) -> pd.Series:
    """Boolean mask of ALL outlier rows (the detect_* helpers truncate indices for display)."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    series = df[column]
    if method == "iqr":
        factor = kwargs.get("factor", 1.5)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return ((series < q1 - factor * iqr) | (series > q3 + factor * iqr)).fillna(False)
    if method == "zscore":
        std = series.std()
        if not std or pd.isna(std):
            return pd.Series(False, index=df.index)
        z = ((series - series.mean()) / std).abs()
        return (z > kwargs.get("threshold", 3.0)).fillna(False)
    raise ValueError(f"Unknown method: '{method}'")


def remove_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", **kwargs
) -> Tuple[pd.DataFrame, str]:
    """Remove outliers from a column."""
    if method == "iqr":
        info = detect_outliers_iqr(df, column, factor=kwargs.get("factor", 1.5))
    elif method == "zscore":
        info = detect_outliers_zscore(df, column, threshold=kwargs.get("threshold", 3.0))
    else:
        raise ValueError(f"Unknown method: '{method}'")

    mask = _outlier_mask(df, column, method, **kwargs)
    removed = int(mask.sum())
    df = df.loc[~mask].reset_index(drop=True)
    return df, f"Removed {removed} outliers from '{column}' using {method}"


def cap_outliers(
    df: pd.DataFrame, column: str, method: str = "iqr", **kwargs
) -> Tuple[pd.DataFrame, str]:
    """Cap outliers (winsorize) instead of removing them."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    if method == "iqr":
        info = detect_outliers_iqr(df, column, factor=kwargs.get("factor", 1.5))
        lower = info["lower_bound"]
        upper = info["upper_bound"]
    elif method == "zscore":
        series = df[column].dropna()
        mean = float(series.mean())
        std = float(series.std())
        threshold = kwargs.get("threshold", 3.0)
        lower = mean - threshold * std
        upper = mean + threshold * std
    else:
        raise ValueError(f"Unknown method: '{method}'")

    capped = int(((df[column] < lower) | (df[column] > upper)).sum())
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df, f"Capped {capped} outliers in '{column}' to [{round(lower, 2)}, {round(upper, 2)}]"


# =============================================================================
# Data Type Correction
# =============================================================================


def detect_type_issues(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect potential data type issues and suggest corrections."""
    issues = []

    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].dropna().head(50)

        if len(sample) == 0:
            continue

        # Numeric stored as string
        if _is_text_dtype(dtype):
            try:
                pd.to_numeric(sample)
                issues.append({
                    "column": col,
                    "current_type": str(dtype),
                    "suggested_type": "numeric",
                    "reason": "Values appear to be numeric stored as text",
                })
                continue
            except (ValueError, TypeError):
                pass

            # Datetime stored as string
            try:
                pd.to_datetime(sample)
                issues.append({
                    "column": col,
                    "current_type": str(dtype),
                    "suggested_type": "datetime",
                    "reason": "Values appear to be dates stored as text",
                })
                continue
            except (ValueError, TypeError):
                pass

    return issues


def convert_column_type(
    df: pd.DataFrame, column: str, target_type: str
) -> Tuple[pd.DataFrame, str]:
    """Convert a column to a target data type."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    original_type = str(df[column].dtype)

    if target_type == "numeric":
        df[column] = pd.to_numeric(df[column], errors="coerce")
    elif target_type == "datetime":
        df[column] = pd.to_datetime(df[column], errors="coerce")
    elif target_type == "string":
        df[column] = df[column].astype(str)
    elif target_type == "category":
        df[column] = df[column].astype("category")
    elif target_type == "integer":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    elif target_type == "float":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    else:
        raise ValueError(f"Unsupported target type: '{target_type}'")

    return df, f"Converted '{column}' from {original_type} to {target_type}"
