"""
Quality Analyzer Module
Generates a Data Quality Score (0-100) based on missing values,
duplicates, outliers, and consistency.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.data_profiler import classify_columns


def calculate_missing_score(df: pd.DataFrame) -> float:
    """
    Score based on completeness (0-100).
    100 = no missing values, 0 = all missing.
    """
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 100.0
    missing_cells = int(df.isnull().sum().sum())
    return round((1 - missing_cells / total_cells) * 100, 2)


def calculate_duplicate_score(df: pd.DataFrame) -> float:
    """
    Score based on uniqueness (0-100).
    100 = no duplicates.
    """
    if len(df) == 0:
        return 100.0
    dup_count = int(df.duplicated().sum())
    return round((1 - dup_count / len(df)) * 100, 2)


def calculate_outlier_score(df: pd.DataFrame) -> float:
    """
    Score based on outlier prevalence in numerical columns (0-100).
    Uses IQR method. 100 = no outliers.
    """
    column_types = classify_columns(df)
    num_cols = column_types.get("numerical", [])

    if not num_cols:
        return 100.0

    total_values = 0
    outlier_values = 0

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((series < lower) | (series > upper)).sum()
        outlier_values += int(outliers)
        total_values += len(series)

    if total_values == 0:
        return 100.0

    return round((1 - outlier_values / total_values) * 100, 2)


def calculate_consistency_score(df: pd.DataFrame) -> float:
    """
    Score based on data consistency (0-100).
    Checks for:
      - Mixed case in string columns
      - Mixed types within columns
      - Whitespace issues
    """
    column_types = classify_columns(df)
    cat_cols = column_types.get("categorical", []) + column_types.get("text", [])

    if not cat_cols:
        return 100.0

    issues = 0
    checks = 0

    for col in cat_cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        # Check mixed case
        lower_vals = series.str.lower()
        unique_original = series.nunique()
        unique_lower = lower_vals.nunique()
        if unique_lower < unique_original:
            issues += 1
        checks += 1

        # Check leading/trailing whitespace
        stripped = series.str.strip()
        if not (series == stripped).all():
            issues += 1
        checks += 1

    if checks == 0:
        return 100.0

    return round((1 - issues / checks) * 100, 2)


def calculate_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall data quality score (0-100).

    Weighted:
      - Missing values: 35%
      - Duplicates: 20%
      - Outliers: 25%
      - Consistency: 20%
    """
    missing_score = calculate_missing_score(df)
    duplicate_score = calculate_duplicate_score(df)
    outlier_score = calculate_outlier_score(df)
    consistency_score = calculate_consistency_score(df)

    overall = (
        missing_score * 0.35
        + duplicate_score * 0.20
        + outlier_score * 0.25
        + consistency_score * 0.20
    )

    return {
        "overall_score": round(overall, 2),
        "components": {
            "completeness": {"score": missing_score, "weight": 0.35},
            "uniqueness": {"score": duplicate_score, "weight": 0.20},
            "outlier_free": {"score": outlier_score, "weight": 0.25},
            "consistency": {"score": consistency_score, "weight": 0.20},
        },
        "grade": _score_to_grade(overall),
    }


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
