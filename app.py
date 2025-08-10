# app.py — Streamlit Cloud friendly orchestrator
import sys
import pathlib
import importlib
import inspect
import pandas as pd
import streamlit as st

# --- Ensure repo root is on PYTHONPATH (helps on Streamlit Cloud if using subfolders) ---
APP_DIR = pathlib.Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# --- Import local modules (reload to avoid stale copies on rerun) ---
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
)

# Optional extras (won’t break if you don’t use them)
# If your visualization.py doesn’t have these yet, comment them out or update that file.
try:
    from visualization import (
        render_quick_filters,
        render_search_box,
        apply_quick_filters_and_search,
        version_badge,  # shows module version/path in sidebar
        __VIS_VERSION__,
    )
    HAS_EXTRAS = True
except Exception:
    HAS_EXTRAS = False
    __VIS_VERSION__ = "unknown"


st.set_page_config(page_title="Financial Stock Screener", layout="wide")


def show_diagnostics():
    """Sidebar diagnostics to prove the correct visualization.py is loaded."""
    st.sidebar.markdown("### Diagnostics")
    if HAS_EXTRAS:
        try:
            version_badge()
        except Exception:
            st.sidebar.caption("version_badge() failed to render.")
    st.sidebar.caption(f"Visualization version: {__VIS_VERSION__}")
    st.sidebar.caption("Imported visualization from:")
    try:
        st.sidebar.code(str(pathlib.Path(visualization.__file__).resolve()))
    except Exception as e:
        st.sidebar.write(f"(path unavailable: {e})")

    # peek first lines of the module so you know which file is active
    try:
        src_head = inspect.getsource(visualization)[:300]
        st.sidebar.code(src_head + "\n...[truncated]")
    except Exception:
        pass

    # cache utilities
    if st.sidebar.button("Force clear cache"):
        st.cache_data.clear()
        st.rerun()


def main():
    st.title("Financial Stock Screener")
    st.caption(
        "Upload an Excel with at least: Symbol, Sector, Industry Group, Industry (Name optional). "
        "Filter in the sidebar. Fetch Yahoo Finance data on demand."
    )

    # Diagnostics (confirm the right module is loaded on Streamlit Cloud)
    show_diagnostics()

    # === Upload ===
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

    # Optional: show raw uploaded data
    checkbox_show_raw(df)

    # === Sidebar filters (taxonomy) ===
    selected_sector, selected_industry_group, selected_industry = render_sidebar_filters(df)

    # === Apply filters (logic) ===
    filtered_df = apply_filters(
        df,
        selected_sector=selected_sector,
        selected_industry_group=selected_industry_group,
        selected_industry=selected_industry,
    )

    # === Optional extras: quick filters + search ===
    if HAS_EXTRAS:
        try:
            quick = render_quick_filters(filtered_df)  # More filters: Theme / Country / Asset_Type
            query = render_search_box()                # Search Symbol/Name
            filtered_df = apply_quick_filters_and_search(filtered_df, quick, query)
        except Exception:
            pass  # Stay resilient if extras are missing

    st.subheader("Filtered Stocks")
    st.write(f"Found **{len(filtered_df)}** rows matching your criteria.")
    if filtered_df.empty:
        st.warning("No rows match your filter criteria.")
        return

    show_filtered_table(filtered_df)

    # === Fetch + merge (on demand) ===
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
                st.warning("No valid symbols found in the filtered table.")
                return

            financial_data = fetch_yfinance_data(symbols)

        if financial_data is None or financial_data.empty:
            st.warning("No financial data was fetched. Please check your symbols or try again later.")
            return

        result_df = merge_results(filtered_df, financial_data)
        show_results_table(result_df)
        download_csv_button(result_df, "stock_screener_results.csv")


if __name__ == "__main__":
    main()
