import sys
import pathlib
import importlib
import pandas as pd
import streamlit as st

# Ensure repo root is on PYTHONPATH (for Streamlit Cloud)
APP_DIR = pathlib.Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import and reload local modules
import data_loader
import analysis
import visualization

importlib.reload(data_loader)
importlib.reload(analysis)
importlib.reload(visualization)

from data_loader import load_data, REQUIRED_COLUMNS, validate_columns
from analysis import apply_filters, fetch_yfinance_data, merge_results
from visualization import (
    checkbox_show_raw,
    render_sidebar_filters,
    show_filtered_table,
    show_results_table,
    download_csv_button,
    render_quick_filters,
    render_search_box,
    apply_quick_filters_and_search,
)

st.set_page_config(page_title="Financial Stock Screener", layout="wide")


def add_numeric_filters(df: pd.DataFrame):
    """Sidebar numeric filters for common metrics."""
    st.sidebar.subheader("Numeric Filters")
    filters = {}

    if "PE Ratio" in df.columns:
        min_pe = float(df["PE Ratio"].min(skipna=True) or 0)
        max_pe = float(df["PE Ratio"].max(skipna=True) or 100)
        filters["PE Ratio"] = st.sidebar.slider("P/E Ratio", min_pe, max_pe, (min_pe, max_pe))

    if "Dividend Yield" in df.columns:
        min_div = float(df["Dividend Yield"].min(skipna=True) or 0)
        max_div = float(df["Dividend Yield"].max(skipna=True) or 10)
        filters["Dividend Yield"] = st.sidebar.slider("Dividend Yield %", min_div, max_div, (min_div, max_div))

    if "Market Cap" in df.columns:
        min_mc = float(df["Market Cap"].min(skipna=True) or 0)
        max_mc = float(df["Market Cap"].max(skipna=True) or 1e12)
        filters["Market Cap"] = st.sidebar.slider(
            "Market Cap ($)", min_mc, max_mc, (min_mc, max_mc)
        )

    return filters


def apply_numeric_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter DataFrame based on numeric slider selections."""
    out = df.copy()
    for col, (low, high) in filters.items():
        if col in out.columns:
            out = out[(out[col] >= low) & (out[col] <= high)]
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add Value, Growth, Momentum scores."""
    out = df.copy()

    # Value Score: low P/E + high Dividend Yield
    if "PE Ratio" in out.columns and "Dividend Yield" in out.columns:
        pe_rank = out["PE Ratio"].rank(ascending=True, pct=True)  # lower P/E = better
        dy_rank = out["Dividend Yield"].rank(ascending=False, pct=True)  # higher yield = better
        out["Value Score"] = ((pe_rank + dy_rank) / 2 * 100).round(1)

    # Growth Score: EPS growth % (if column exists)
    if "EPS Growth %" in out.columns:
        out["Growth Score"] = out["EPS Growth %"].rank(ascending=False, pct=True) * 100

    # Momentum Score: 52-week percentile
    if "Current Price" in out.columns and "52 Week Low" in out.columns and "52 Week High" in out.columns:
        rng = out["52 Week High"] - out["52 Week Low"]
        rng = rng.replace(0, pd.NA)
        out["Momentum Score"] = ((out["Current Price"] - out["52 Week Low"]) / rng * 100).round(1)

    return out


def main():
    st.title("Financial Stock Screener (Pro)")
    st.caption(
        "Upload an Excel file with Symbol, Sector, Industry Group, Industry (Name optional). "
        "Filter in the sidebar. Fetch Yahoo Finance data for scores and metrics."
    )

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if not uploaded_file:
        st.info("Please upload an Excel file to begin.")
        return

    df = load_data(uploaded_file)
    if df is None or df.empty:
        st.error("Could not read the file or it was empty.")
        return

    if not validate_columns(df):
        st.error(f"Missing required columns. Your file must include: {sorted(REQUIRED_COLUMNS)}")
        st.stop()

    checkbox_show_raw(df)

    selected_sector, selected_industry_group, selected_industry = render_sidebar_filters(df)

    filtered_df = apply_filters(
        df,
        selected_sector=selected_sector,
        selected_industry_group=selected_industry_group,
        selected_industry=selected_industry,
    )

    quick = render_quick_filters(filtered_df)
    query = render_search_box()
    filtered_df = apply_quick_filters_and_search(filtered_df, quick, query)

    st.subheader("Filtered Stocks (Before Metrics)")
    st.write(f"Found **{len(filtered_df)}** rows matching your criteria.")
    if filtered_df.empty:
        st.warning("No rows match your filter criteria.")
        return

    show_filtered_table(filtered_df)

    if st.button("Fetch Financial Data from Yahoo Finance"):
        with st.spinner("Fetching data from Yahoo Finance…"):
            symbols = (
                filtered_df["Symbol"]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            if not symbols:
                st.warning("No valid symbols found.")
                return

            financial_data = fetch_yfinance_data(symbols)

        if financial_data is None or financial_data.empty:
            st.warning("No financial data was fetched.")
            return

        result_df = merge_results(filtered_df, financial_data)

        # === Add scores ===
        result_df = add_scores(result_df)

        # === Numeric filters ===
        num_filters = add_numeric_filters(result_df)
        result_df = apply_numeric_filters(result_df, num_filters)

        # === Sort by score ===
        score_cols = [c for c in ["Value Score", "Growth Score", "Momentum Score"] if c in result_df.columns]
        if score_cols:
            sort_choice = st.sidebar.selectbox("Sort by Score", ["None"] + score_cols)
            if sort_choice != "None":
                result_df = result_df.sort_values(sort_choice, ascending=False)

        show_results_table(result_df)
        download_csv_button(result_df, "stock_screener_results.csv")


if __name__ == "__main__":
    main()
