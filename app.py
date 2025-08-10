import streamlit as st
import pandas as pd

from data_loader import load_data, REQUIRED_COLUMNS, validate_columns
from analysis import apply_filters, fetch_yfinance_data, merge_results
from visualization import (
    checkbox_show_raw,
    render_sidebar_filters,
    show_filtered_table,
    show_results_table,
    download_csv_button,
)

st.set_page_config(page_title="Financial Stock Screener", layout="wide")


def main():
    st.title("Financial Stock Screener")
    st.write(
        "Upload an Excel file with stock symbols and use filters to screen stocks. "
        "Data will only be fetched from Yahoo Finance after applying filters."
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
        st.error(f"Missing required columns. Your file needs these columns: {REQUIRED_COLUMNS}")
        return

    # Optional: show raw uploaded data
    checkbox_show_raw(df)

    # Sidebar filters
    selected_sector, selected_industry_group, selected_industry = render_sidebar_filters(df)

    # Apply filters
    filtered_df = apply_filters(df, selected_sector, selected_industry_group, selected_industry)

    st.subheader("Filtered Stocks")
    st.write(f"Found {len(filtered_df)} stocks matching your criteria")

    if filtered_df.empty:
        st.warning("No stocks match your filter criteria.")
        return

    show_filtered_table(filtered_df)

    if st.button("Fetch Financial Data from Yahoo Finance"):
        with st.spinner("Fetching data from Yahoo Finance. This may take a while..."):
            symbols = filtered_df["Symbol"].dropna().astype(str).unique().tolist()
            financial_data = fetch_yfinance_data(symbols)

        if financial_data.empty:
            st.warning("No financial data was fetched. Please check your symbols.")
            return

        result_df = merge_results(filtered_df, financial_data)
        show_results_table(result_df)
        download_csv_button(result_df, "stock_screener_results.csv")


if __name__ == "__main__":
    main()
