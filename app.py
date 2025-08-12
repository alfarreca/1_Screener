import streamlit as st
import pandas as pd
import numpy as np

import data_loader
import visualization
import analysis


# ---------- Numeric safety ----------
def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------- Filters ----------
def add_numeric_filters(df: pd.DataFrame):
    filters = {}
    st.sidebar.header("Numeric Filters")

    def safe_slider(label, colname):
        if colname in df.columns:
            col_data = as_num(df[colname])
            if col_data.notna().any():
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
                    filters[colname] = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))
    # Example filters
    safe_slider("P/E Ratio", "PE Ratio")
    safe_slider("Dividend Yield (%)", "Dividend Yield")
    safe_slider("Market Cap", "Market Cap")

    return filters


def apply_numeric_filters(df: pd.DataFrame, filters):
    out = df.copy()
    for col, (low, high) in filters.items():
        out = out[(as_num(out[col]) >= low) & (as_num(out[col]) <= high)]
    return out


# ---------- Scores ----------
def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Value Score: inverse rank of PE Ratio
    if "PE Ratio" in out.columns:
        pe_rank = as_num(out["PE Ratio"]).rank(ascending=True, pct=True)
        out["Value Score"] = (1 - pe_rank) * 100

    # Growth Score: inverse rank of PEG Ratio
    if "PEG Ratio" in out.columns:
        peg_rank = as_num(out["PEG Ratio"]).rank(ascending=True, pct=True)
        out["Growth Score"] = (1 - peg_rank) * 100

    # Momentum Score: 52-week percentile (fixed to keep float dtype)
    if {"Current Price", "52 Week Low", "52 Week High"} <= set(out.columns):
        cp = as_num(out["Current Price"])
        lo = as_num(out["52 Week Low"])
        hi = as_num(out["52 Week High"])

        rng = (hi - lo)  # stays float dtype
        rng = rng.replace(0, np.nan)

        momentum = ((cp - lo) / rng) * 100
        out["Momentum Score"] = momentum.astype(float).round(1)

    # Value-Contrarian Score: strong value + weak/mid momentum
    if "Value Score" in out.columns and "Momentum Score" in out.columns:
        val = as_num(out["Value Score"])
        mom = as_num(out["Momentum Score"])
        mask_mid_low = mom <= 60
        out["Value-Contrarian Score"] = (val * mask_mid_low).fillna(0).round(1)

    return out


# ---------- Main ----------
def main():
    st.set_page_config(page_title="Stock Screener", layout="wide")

    st.title("📊 Professional Stock Screener")

    # 1. Load data
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Upload a file to start.")
        return

    df = data_loader.load_data(uploaded_file)
    visualization.checkbox_show_raw(df)

    # 2. Filters
    selected_sector, selected_ig, selected_industry = visualization.render_sidebar_filters(df)
    quick_filters = visualization.render_quick_filters(df)
    search_term = visualization.render_search_box()
    num_filters = add_numeric_filters(df)

    # 3. Apply filters
    filtered_df = df.copy()
    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df["Sector"] == selected_sector]
    if selected_ig != "All":
        filtered_df = filtered_df[filtered_df["Industry Group"] == selected_ig]
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry"] == selected_industry]

    filtered_df = visualization.apply_quick_filters_and_search(filtered_df, quick_filters, search_term)
    filtered_df = apply_numeric_filters(filtered_df, num_filters)

    # 4. Preview filtered data
    visualization.show_filtered_table(filtered_df)

    # 5. Analysis
    result_df = analysis.fetch_yahoo_metrics(filtered_df)
    result_df = add_scores(result_df)

    # 6. Show results
    visualization.show_results_table(result_df)

    # 7. Downloads
    visualization.download_csv_button(result_df, filename="screener_results.csv")


if __name__ == "__main__":
    main()
