import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List


def apply_filters(
    df: pd.DataFrame,
    selected_sector: str,
    selected_industry_group: str,
    selected_industry: str,
) -> pd.DataFrame:
    """Filter the input dataframe by the selected taxonomy values."""
    out = df.copy()

    if selected_sector != "All":
        out = out[out["Sector"] == selected_sector]

    if selected_industry_group != "All":
        out = out[out["Industry Group"] == selected_industry_group]

    if selected_industry != "All":
        out = out[out["Industry"] == selected_industry]

    return out


@st.cache_data(show_spinner=False)
def fetch_yfinance_data(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch key metrics from Yahoo Finance for a list of symbols.
    Uses yf.Ticker().info; wrapped in try/except to avoid hard failures.
    """
    if not symbols:
        return pd.DataFrame()

    # Dates are not directly used for .info, but left here if you expand later
    _ = datetime.now()
    _ = _ - timedelta(days=365)

    results = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            results.append(
                {
                    "Symbol": sym,
                    "Current Price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "52 Week High": info.get("fiftyTwoWeekHigh"),
                    "52 Week Low": info.get("fiftyTwoWeekLow"),
                    "PE Ratio": info.get("trailingPE"),
                    "Market Cap": info.get("marketCap"),
                    "Dividend Yield": (info.get("dividendYield") * 100) if info.get("dividendYield") else None,
                    "Beta": info.get("beta"),
                    "Volume": info.get("volume"),
                    "Avg Volume": info.get("averageVolume"),
                    "Sector (YF)": info.get("sector", "N/A"),
                    "Industry (YF)": info.get("industry", "N/A"),
                }
            )
        except Exception:
            # Soft-fail a single symbol but continue
            st.warning(f"Could not fetch data for {sym}")
            continue

    return pd.DataFrame(results)


def merge_results(filtered_df: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
    """Merge the original filtered rows with fetched Yahoo metrics."""
    result_df = pd.merge(filtered_df, financial_data, on="Symbol", how="left")
    # Keep a clean column order (optional)
    front_cols = [c for c in ["Symbol", "Name", "Sector", "Industry Group", "Industry"] if c in result_df.columns]
    other_cols = [c for c in result_df.columns if c not in front_cols]
    return result_df[front_cols + other_cols]
