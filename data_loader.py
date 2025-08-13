# data_loader.py
from __future__ import annotations

import io
import re
from typing import Optional, Set, Dict, List
import pandas as pd

# Required columns as per your spec
REQUIRED_COLUMNS: Set[str] = {"Symbol", "Name", "Country", "Asset_Type", "Notes"}

# Common header aliases we normalize
HEADER_ALIASES: Dict[str, str] = {
    "ticker": "Symbol",
    "tickers": "Symbol",
    "symbol": "Symbol",
    "company": "Name",
    "security": "Name",
    "description": "Name",
    "asset type": "Asset_Type",
    "asset-type": "Asset_Type",
    "asset": "Asset_Type",
    "note": "Notes",
}

# Quick test for likely ticker strings (e.g., AAPL, BRK.B, RIO, 7203.T)
_TICKER_RX = re.compile(r"^[A-Z0-9]{1,5}([.\-][A-Z0-9]{1,4})?$")


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols: List[str] = []
    for c in df.columns:
        raw = str(c).strip()
        key = re.sub(r"\s+", " ", raw).lower()
        new_cols.append(HEADER_ALIASES.get(key, raw))
    df.columns = new_cols
    return df


def _try_recover_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """If 'Symbol' is missing, try to recover from Unnamed:* or alias columns."""
    df = df.copy()
    if "Symbol" in df.columns:
        return df

    # 1) Check for any column that looks like tickers
    for c in list(df.columns):
        series = df[c].astype(str).str.strip().str.upper()
        if series.notna().any():
            sample = series.dropna().head(50)
            if len(sample) > 0 and (sample.apply(lambda x: bool(_TICKER_RX.match(x))).mean() > 0.7):
                df.rename(columns={c: "Symbol"}, inplace=True)
                return df

    # 2) Specific check for Unnamed:* columns (Excel index columns)
    for c in df.columns:
        if str(c).lower().startswith("unnamed:"):
            series = df[c].astype(str).str.strip().str.upper()
            sample = series.dropna().head(50)
            if len(sample) > 0 and (sample.apply(lambda x: bool(_TICKER_RX.match(x))).mean() > 0.7):
                df.rename(columns={c: "Symbol"}, inplace=True)
                return df

    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV/XLSX, normalize headers, recover 'Symbol' if it came in as 'Unnamed:*',
    clean whitespace, upper-case symbols, and ensure required columns exist.
    """
    if uploaded_file is None:
        return None

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        try:
            content = uploaded_file.read()
            if name.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception:
            return None

    # Normalize & recover
    df = _normalize_headers(df)
    df = _try_recover_symbol_column(df)
    df = _ensure_required_columns(df)

    # Clean values
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()
    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str).str.strip()
    if "Asset_Type" in df.columns:
        df["Asset_Type"] = df["Asset_Type"].astype(str).str.strip()
    if "Notes" in df.columns:
        # keep as string, allow "None"
        df["Notes"] = df["Notes"].astype(str).str.strip()

    # Drop rows without a valid symbol
    df = df[df["Symbol"].astype(str).str.len() > 0].reset_index(drop=True)

    # Remove empty unnamed columns that slip through
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed:")]

    return df


def validate_columns(df: pd.DataFrame) -> bool:
    """Check that the uploaded file has the 5 required columns (after auto-fix)."""
    if df is None or df.empty:
        return False
    return REQUIRED_COLUMNS.issubset(set(df.columns))
