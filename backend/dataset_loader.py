"""
Dataset Loader Module
Handles loading datasets from CSV, Excel (.xlsx), TSV, and JSON formats.
Dynamically detects columns and data types without any hardcoded assumptions.
"""

import io
import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".json"}


def detect_file_type(filename: str) -> str:
    """Detect file type from extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mapping = {
        "csv": "csv",
        "tsv": "tsv",
        "xlsx": "excel",
        "xls": "excel",
        "json": "json",
    }
    return mapping.get(ext, "unknown")


def load_dataset(
    file_content: bytes,
    filename: str,
    encoding: str = "utf-8",
    separator: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a dataset from raw file bytes.

    Args:
        file_content: Raw bytes of the uploaded file.
        filename: Original filename (used to detect format).
        encoding: Character encoding (default utf-8).
        separator: Optional custom separator for CSV/TSV.

    Returns:
        Tuple of (DataFrame, metadata dict).
    """
    file_type = detect_file_type(filename)

    if file_type == "csv":
        sep = separator or ","
        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, sep=sep)
    elif file_type == "tsv":
        sep = separator or "\t"
        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, sep=sep)
    elif file_type == "excel":
        df = pd.read_excel(io.BytesIO(file_content))
    elif file_type == "json":
        data = json.loads(file_content.decode(encoding))
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. "
            f"Supported: CSV, Excel (.xlsx), TSV, JSON"
        )

    metadata = {
        "filename": filename,
        "file_type": file_type,
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "columns": list(df.columns),
        "encoding": encoding,
    }

    return df, metadata


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download."""
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue()


def dataframe_to_json_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to JSON bytes for download."""
    return df.to_json(orient="records", indent=2).encode("utf-8")
