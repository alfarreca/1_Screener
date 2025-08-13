# data_loader.py
from __future__ import annotations

import io
from typing import Optional, Set
import pandas as pd

# New required columns per your spec
REQUIRED_COLUMNS: Set[str] = {"Symbol", "Name", "Country", "Asset_Type", "Notes"}


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Trim spaces and standardize header names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel. We expect at least the five required columns.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            # default to Excel
            # If the workbook has multiple sheets, take the first sheet
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        # Sometimes Streamlit gives a BytesIO-like object; try bytes then read_excel
        try:
            content = uploaded_file.read()
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception:
            return None

    df = _normalize_headers(df)

    # Ensure at least the required columns exist (others are optional and will be filled by Yahoo)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Create missing required columns as empty (so downstream never KeyErrors)
        for c in missing:
            df[c] = pd.NA

    # Keep only relevant columns plus any extras the user included
    return df


def validate_columns(df: pd.DataFrame) -> bool:
    """Check that the uploaded file has the 5 required columns."""
    if df is None or df.empty:
        return False
    have = set(df.columns)
    return REQUIRED_COLUMNS.issubset(have)
