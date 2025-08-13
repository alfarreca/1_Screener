# app.py
import numpy as np
import pandas as pd
import streamlit as st

import data_loader
import visualization
import analysis


# ---------------- Helpers ----------------
def as_num(s: pd.Series) -> pd.Series:
    """Force numeric (coerce errors to NaN)."""
    return pd.to_numeric(s, errors="coerce")


# ---------------- Sidebar: Numeric Filters ----------------
def add_numeric_filters(df: pd.DataFrame) -> dict:
    """
    Build safe sliders only when a column has a real numeric range.
    Returns a dict: {col: (low, high)}.
    """
    st.sidebar.subheader("Numeric Filters")
    filters: dict = {}

    def safe_slider(col: str, label: str):
        if col not in df.columns:
            return
        nums = as_num(df[col]).replace([np.inf, -np.inf], np.nan).dropna()
        if nums.empty:
            return
        lo, hi = float(nums.min()), float(nums.max())
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            filters[col] = st.sidebar.slider(label, lo, hi, (lo, hi))

    safe_slider("PE Ratio", "P/E Ratio")
    safe_slider("Dividend Yield", "Dividend Yield %")
    safe_slider("Market Cap", "Market Cap ($)")
    return filters


def apply_numeric_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    for col, (low, high) in filters.items():
        nums = as_num(out[col])
        out = out[(nums >= low) & (nums <= high)]
    return out


# ---------------- Scoring ----------------
def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Value, Growth, Momentum, and Value-Contrarian scores.
    All scores in [0,100].
    """
    out = df.copy()

    # Value Score: blend of low P/E (good) + high Dividend Yield (good)
    if {"PE Ratio", "Dividend Yield"} <= set(out.columns):
        pe = as_num(out["PE Ratio"])
        dy = as_num(out["Dividend Yield"])
        if pe.notna().any() and dy.notna().any():
            pe_rank = pe.rank(ascending=True, pct=True)          # lower P/E = better
            dy_rank = dy.rank(ascending=False, pct=True)         # higher Yield = better
            out["Value Score"] = ((pe_rank + dy_rank) / 2 * 100).round(1)

    # Growth Score (optional): if you later add EPS Growth %
    if "EPS Growth %" in out.columns:
        eps = as_num(out["EPS Growth %"])
        if eps.notna().any():
            out["Growth Score"] = (eps.rank(ascending=False, pct=True) * 100).round(1)

    # Momentum Score: position within 52-week range (float-safe)
    if {"Current Price", "52 Week Low", "52 Week High"} <= set(out.columns):
        cp = as_num(out["Current Price"])
        lo = as_num(out["52 Week Low"])
        hi = as_num(out["52 Week High"])
        rng = (hi - lo)             # stays float dtype
        rng = rng.replace(0, np.nan)
        momentum = (cp - lo) / rng * 100
        out["Momentum Score"] = momentum.astype(float).round(1)

    # Value-Contrarian Score: high value + mid/low momentum (sweet spot ~40%)
    if {"Value Score", "Momentum Score"} <= set(out.columns):
        v = as_num(out["Value Score"]).clip(0, 100) / 100.0
        m = as_num(out["Momentum Score"]).clip(0, 100) / 100.0
        m_target = 0.40   # prefer not-too-hot names
        falloff = 0.40
        contrarian = 1.0 - (np.abs(m - m_target) / falloff)
        contrarian = np.clip(contrarian, 0.0, 1.0)
        out["Value-Contrarian Score"] = (v * contrarian * 100).round(1)

    return out


# ---------------- App ----------------
def main():
    st.set_page_config(page_title="Stock Screener (Pro, Dark)", layout="wide")
    st.title("📊 Professional Stock Screener")

    # Upload
    uploaded = st.file_uploader("Upload Excel/CSV with columns: Symbol, Sector, Industry Group, Industry (Name optional)",
                                type=["xlsx", "xls", "csv"])
    if not uploaded:
        st.info("Upload a file to begin.")
        return

    # Load
    df = data_loader.load_data(uploaded)
    if df is None or df.empty:
        st.error("Could not read the file or it was empty.")
        return

    # Validate minimal columns
    if not data_loader.validate_columns(df):
        missing = sorted(data_loader.REQUIRED_COLUMNS)
        st.error(f"Missing required columns. Your file must include: {missing}")
        return

    # Optional raw view
    visualization.checkbox_show_raw(df)

    # Sidebar filters (taxonomy)
    sel_sector, sel_ig, sel_ind = visualization.render_sidebar_filters(df)

    # Apply taxonomy filters (server-side)
    filtered_df = analysis.apply_filters(
        df,
        selected_sector=sel_sector,
        selected_industry_group=sel_ig,
        selected_industry=sel_ind,
    )

    # Quick filters + search (Theme/Country/Asset_Type + Symbol/Name)
    quick = visualization.render_quick_filters(filtered_df)
    query = visualization.render_search_box()
    filtered_df = visualization.apply_quick_filters_and_search(filtered_df, quick, query)

    # Show preview
    st.subheader("Filtered Stocks (before metrics)")
    st.caption(f"{len(filtered_df)} rows match your filters.")
    if filtered_df.empty:
        st.warning("No rows match your filters. Adjust and try again.")
        return
    visualization.show_filtered_table(filtered_df)

    # Fetch metrics
    if st.button("Fetch Financial Data from Yahoo Finance"):
        with st.spinner("Fetching metrics…"):
            symbols = (
                filtered_df["Symbol"]
                .astype(str).str.strip().str.upper()
                .replace("", np.nan).dropna().unique().tolist()
            )
            if not symbols:
                st.warning("No valid symbols to fetch.")
                return

            fin = analysis.fetch_yfinance_data(symbols)

        # Consolidated warning for failures
        failed = getattr(fin, "attrs", {}).get("failed_symbols", [])
        if failed:
            st.warning(f"Could not fetch data for: {', '.join(failed)}")

        # Merge & score
        result_df = analysis.merge_results(filtered_df, fin)
        result_df = add_scores(result_df)

        # Numeric sliders (work on the results)
        sliders = add_numeric_filters(result_df)
        result_df = apply_numeric_filters(result_df, sliders)

        # Sort by any score
        score_cols = [c for c in ["Value Score", "Growth Score", "Momentum Score", "Value-Contrarian Score"]
                      if c in result_df.columns]
        if score_cols:
            choice = st.sidebar.selectbox("Sort by Score", ["None"] + score_cols, index=0)
            if choice != "None":
                result_df = result_df.sort_values(choice, ascending=False, na_position="last")

        # Show results (with progress bars + sparkline from visualization.py)
        visualization.show_results_table(result_df)

        # Downloads
        visualization.download_csv_button(result_df, filename="screener_results.csv")


if __name__ == "__main__":
    main()
