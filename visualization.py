import streamlit as st
import pandas as pd
from typing import Tuple, Optional


def checkbox_show_raw(df: pd.DataFrame) -> None:
    """Optional toggle to display raw uploaded data."""
    if st.checkbox("Show raw data"):
        st.dataframe(df, use_container_width=True)


def render_sidebar_filters(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Build sidebar dropdowns and return selected values."""
    st.sidebar.header("Filters")

    sectors = ["All"] + sorted(df["Sector"].dropna().astype(str).unique().tolist())
    selected_sector = st.sidebar.selectbox("Select Sector", sectors, index=0)

    if selected_sector != "All":
        mask = df["Sector"] == selected_sector
        ig_list = ["All"] + sorted(df.loc[mask, "Industry Group"].dropna().astype(str).unique().tolist())
    else:
        ig_list = ["All"] + sorted(df["Industry Group"].dropna().astype(str).unique().tolist())
    selected_industry_group = st.sidebar.selectbox("Select Industry Group", ig_list, index=0)

    if selected_sector != "All" and selected_industry_group != "All":
        mask = (df["Sector"] == selected_sector) & (df["Industry Group"] == selected_industry_group)
        ind_list = ["All"] + sorted(df.loc[mask, "Industry"].dropna().astype(str).unique().tolist())
    elif selected_sector != "All":
        mask = df["Sector"] == selected_sector
        ind_list = ["All"] + sorted(df.loc[mask, "Industry"].dropna().astype(str).unique().tolist())
    elif selected_industry_group != "All":
        mask = df["Industry Group"] == selected_industry_group
        ind_list = ["All"] + sorted(df.loc[mask, "Industry"].dropna().astype(str).unique().tolist())
    else:
        ind_list = ["All"] + sorted(df["Industry"].dropna().astype(str).unique().tolist())
    selected_industry = st.sidebar.selectbox("Select Industry", ind_list, index=0)

    return selected_sector, selected_industry_group, selected_industry


def show_filtered_table(filtered_df: pd.DataFrame) -> None:
    """Display a compact table of the filtered rows."""
    cols = [c for c in ["Symbol", "Name", "Sector", "Industry Group", "Industry"] if c in filtered_df.columns]
    st.dataframe(filtered_df[cols], use_container_width=True)


def show_results_table(result_df: pd.DataFrame) -> None:
    """Display results with financial metrics."""
    st.subheader("Results with Yahoo Finance Metrics")
    st.dataframe(result_df, use_container_width=True)


def download_csv_button(df: pd.DataFrame, filename: str = "results.csv") -> Optional[bytes]:
    """Render a CSV download button and return the CSV bytes (if needed elsewhere)."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )
    return csv_bytes
