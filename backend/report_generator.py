"""
Report Generator Module
Generates AI-powered recommendations for data cleaning, identifies issues,
and suggests preprocessing techniques.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.data_profiler import classify_columns, compute_missing_values
from backend.quality_analyzer import calculate_quality_score


def generate_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Generate AI-powered recommendations for cleaning and preprocessing.

    Categories:
      - missing_values: Columns needing imputation
      - outliers: Columns with potential outliers
      - encoding: Categorical columns needing encoding
      - scaling: Numerical columns that may benefit from scaling
      - type_correction: Columns with incorrect types
      - high_cardinality: Categorical columns with too many unique values
    """
    recommendations = []
    column_types = classify_columns(df)
    missing = compute_missing_values(df)

    # --- Missing Value Recommendations ---
    for col, info in missing.items():
        if info["percentage"] > 0:
            priority = "high" if info["percentage"] > 30 else ("medium" if info["percentage"] > 10 else "low")

            if col in column_types.get("numerical", []):
                method = "median" if info["percentage"] > 20 else "mean"
                reason = f"{info['percentage']}% missing — use {method} imputation for numerical data"
            elif col in column_types.get("categorical", []):
                method = "mode"
                reason = f"{info['percentage']}% missing — use mode imputation for categorical data"
            else:
                method = "ffill"
                reason = f"{info['percentage']}% missing — consider forward fill or dropping"

            if info["percentage"] > 60:
                method = "drop_column"
                reason = f"{info['percentage']}% missing — consider dropping this column entirely"

            recommendations.append({
                "category": "missing_values",
                "column": col,
                "priority": priority,
                "suggestion": method,
                "reason": reason,
            })

    # --- Outlier Recommendations ---
    for col in column_types.get("numerical", []):
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

        if outlier_count > 0:
            pct = round((outlier_count / len(series)) * 100, 1)
            priority = "high" if pct > 10 else ("medium" if pct > 5 else "low")
            recommendations.append({
                "category": "outliers",
                "column": col,
                "priority": priority,
                "suggestion": "cap_outliers" if pct < 10 else "remove_outliers",
                "reason": f"{outlier_count} outliers ({pct}%) detected via IQR method",
            })

    # --- Encoding Recommendations ---
    for col in column_types.get("categorical", []):
        nunique = df[col].nunique()
        if nunique == 2:
            recommendations.append({
                "category": "encoding",
                "column": col,
                "priority": "low",
                "suggestion": "label_encode",
                "reason": f"Binary column ({nunique} unique values) — label encoding recommended",
            })
        elif nunique <= 10:
            recommendations.append({
                "category": "encoding",
                "column": col,
                "priority": "low",
                "suggestion": "one_hot_encode",
                "reason": f"Low cardinality ({nunique} unique) — one-hot encoding recommended",
            })
        elif nunique > 10:
            recommendations.append({
                "category": "high_cardinality",
                "column": col,
                "priority": "medium",
                "suggestion": "label_encode",
                "reason": f"High cardinality ({nunique} unique) — label encoding or target encoding",
            })

    # --- Scaling Recommendations ---
    for col in column_types.get("numerical", []):
        series = df[col].dropna()
        if len(series) < 5:
            continue
        range_val = float(series.max() - series.min())
        skew = float(series.skew())

        if abs(skew) > 2:
            recommendations.append({
                "category": "scaling",
                "column": col,
                "priority": "medium",
                "suggestion": "log_transform",
                "reason": f"Highly skewed (skewness={round(skew, 2)}) — log transform recommended",
            })
        elif range_val > 1000:
            recommendations.append({
                "category": "scaling",
                "column": col,
                "priority": "low",
                "suggestion": "standard_scale",
                "reason": f"Large range ({round(range_val, 1)}) — standardization recommended",
            })

    # --- Type Correction Recommendations ---
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            try:
                pd.to_numeric(sample)
                recommendations.append({
                    "category": "type_correction",
                    "column": col,
                    "priority": "high",
                    "suggestion": "convert_to_numeric",
                    "reason": "Column contains numeric values stored as text",
                })
            except (ValueError, TypeError):
                pass

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return recommendations


def generate_cleaning_report(df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive cleaning report comparing original and cleaned data.
    """
    quality_before = calculate_quality_score(original_df)
    quality_after = calculate_quality_score(df)

    return {
        "summary": {
            "rows_before": len(original_df),
            "rows_after": len(df),
            "columns_before": len(original_df.columns),
            "columns_after": len(df.columns),
            "rows_removed": len(original_df) - len(df),
            "columns_added": max(0, len(df.columns) - len(original_df.columns)),
        },
        "quality_before": quality_before,
        "quality_after": quality_after,
        "quality_improvement": round(
            quality_after["overall_score"] - quality_before["overall_score"], 2
        ),
    }
