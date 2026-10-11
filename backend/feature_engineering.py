"""
Feature Engineering Module
Supports Label Encoding, One-Hot Encoding, Scaling, Log Transform,
and Datetime Feature Extraction — all without requiring fixed column names.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Encoding
# =============================================================================


def label_encode(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply label encoding to a categorical column.
    Returns the modified DataFrame and the mapping used.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    unique_vals = df[column].dropna().unique()
    mapping = {val: idx for idx, val in enumerate(sorted(unique_vals, key=str))}
    df[f"{column}_encoded"] = df[column].map(mapping)

    return df, {
        "column": column,
        "new_column": f"{column}_encoded",
        "mapping": {str(k): v for k, v in mapping.items()},
        "n_classes": len(mapping),
    }


def one_hot_encode(
    df: pd.DataFrame, column: str, drop_first: bool = False, max_categories: int = 20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply one-hot encoding to a categorical column.
    Limits categories to max_categories to avoid explosion.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    source = df[column]
    if source.nunique() > max_categories:
        top = source.value_counts().head(max_categories).index.tolist()
        source = source.where(source.isin(top) | source.isna(), other="Other")

    dummies = pd.get_dummies(source, prefix=column, drop_first=drop_first).astype(int)
    new_columns = list(dummies.columns)
    df = pd.concat([df, dummies], axis=1)

    return df, {
        "column": column,
        "new_columns": new_columns,
        "drop_first": drop_first,
        "n_categories": len(new_columns),
    }


# =============================================================================
# Scaling / Normalization
# =============================================================================


def min_max_scale(
    df: pd.DataFrame, column: str, feature_range: Tuple[float, float] = (0, 1)
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply Min-Max scaling to a numerical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    series = df[column].dropna()
    min_val = float(series.min())
    max_val = float(series.max())

    if max_val == min_val:
        df[f"{column}_scaled"] = 0.0
    else:
        low, high = feature_range
        df[f"{column}_scaled"] = low + (df[column] - min_val) / (max_val - min_val) * (high - low)

    return df, {
        "column": column,
        "new_column": f"{column}_scaled",
        "min": min_val,
        "max": max_val,
        "feature_range": list(feature_range),
    }


def standard_scale(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply Standard (Z-score) scaling to a numerical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    series = df[column].dropna()
    mean = float(series.mean())
    std = float(series.std())

    if std == 0:
        df[f"{column}_standardized"] = 0.0
    else:
        df[f"{column}_standardized"] = (df[column] - mean) / std

    return df, {
        "column": column,
        "new_column": f"{column}_standardized",
        "mean": round(mean, 4),
        "std": round(std, 4),
    }


# =============================================================================
# Transformations
# =============================================================================


def log_transform(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply log transformation (log1p) to handle zero values."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    min_val = float(df[column].min())
    if min_val < 0:
        # Shift so the minimum becomes 0, then log1p
        df[f"{column}_log"] = np.log1p(df[column] - min_val)
    else:
        df[f"{column}_log"] = np.log1p(df[column])

    return df, {
        "column": column,
        "new_column": f"{column}_log",
        "min_original": min_val,
        "shifted": min_val < 0,
    }


# =============================================================================
# Datetime Feature Extraction
# =============================================================================


def extract_datetime_features(
    df: pd.DataFrame, column: str, features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extract features from a datetime column.

    Available features: year, month, day, hour, minute, dayofweek, quarter,
                       is_weekend, day_name, month_name
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors="coerce")

    if features is None:
        features = ["year", "month", "day", "dayofweek", "quarter", "is_weekend"]

    new_columns = []

    for feat in features:
        col_name = f"{column}_{feat}"
        if feat == "year":
            df[col_name] = df[column].dt.year
        elif feat == "month":
            df[col_name] = df[column].dt.month
        elif feat == "day":
            df[col_name] = df[column].dt.day
        elif feat == "hour":
            df[col_name] = df[column].dt.hour
        elif feat == "minute":
            df[col_name] = df[column].dt.minute
        elif feat == "dayofweek":
            df[col_name] = df[column].dt.dayofweek
        elif feat == "quarter":
            df[col_name] = df[column].dt.quarter
        elif feat == "is_weekend":
            df[col_name] = df[column].dt.dayofweek.isin([5, 6]).astype(int)
        elif feat == "day_name":
            df[col_name] = df[column].dt.day_name()
        elif feat == "month_name":
            df[col_name] = df[column].dt.month_name()
        else:
            continue
        new_columns.append(col_name)

    return df, {
        "column": column,
        "features_extracted": features,
        "new_columns": new_columns,
    }


def get_available_operations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return available feature engineering operations based on column types.
    """
    operations = {
        "encoding": [],
        "scaling": [],
        "transformation": [],
        "datetime_extraction": [],
    }

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            operations["scaling"].append({
                "column": col,
                "available": ["min_max_scale", "standard_scale", "log_transform"],
            })
            operations["transformation"].append({
                "column": col,
                "available": ["log_transform"],
            })
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            nunique = df[col].nunique()
            ops = ["label_encode"]
            if nunique <= 20:
                ops.append("one_hot_encode")
            operations["encoding"].append({
                "column": col,
                "available": ops,
                "n_unique": nunique,
            })

        if pd.api.types.is_datetime64_any_dtype(dtype):
            operations["datetime_extraction"].append({
                "column": col,
                "available": ["year", "month", "day", "hour", "dayofweek", "quarter", "is_weekend"],
            })

    return operations
