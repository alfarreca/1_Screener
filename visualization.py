import io
from typing import Tuple, Optional, List, Dict

import pandas as pd
import streamlit as st


# ---------- Small helpers ----------

def _sorted_unique(series: pd.Series) -> List[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def _frontload_columns(df: pd.DataFrame, front: List[str]) -> List[str]:
    front = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return front + rest


# ---------- Raw data toggle ----------

def checkbox_show_raw(df: pd.DataFrame) -> None:
    """Optional toggle to display raw uploaded data."""
    if st.checkbox("Show raw uploaded data"):
        with st.expander("Raw data", expanded=False):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )


# ---------- Sidebar filters ----------

def render_sidebar_filters(df: pd.DataFrame) -> Tuple[str, str, str]:
    st.sidebar.header("Filters")

    sectors = ["All"] + _sorted_unique(df["Sector"])
    selected_sector = st.sidebar.selectbox("Sector", sectors, index=0, key="sector")

    if selected_sector != "All":
        ig_list = ["All"] + _sorted_unique(df.loc[df["Sector"] == selected_sector, "Industry Group"])
    else:
        ig_list = ["All"] + _sorted_unique(df["Industry Group"])
    selected_industry_group = st.sidebar.selectbox("Industry Group", ig_list, index=0, key="ig")

    if selected_sector != "All" and selected_industry_group != "All":
        mask = (df["Sector"] == selected_sector) & (df["Industry Group"] == selected_industry_group)
        ind_list = ["All"] + _sorted_unique(df.loc[mask, "Industry"])
    elif selected_sector != "All":
        mask = df["Sector"] == selected_sector
        ind_list = ["All"] + _sorted_unique(df.loc[mask, "Industry"])
    elif selected_industry_group != "All":
        mask = df["Industry Group"] == selected_industry_group
        ind_list = ["All"] + _sorted_unique(df.loc[mask, "Industry"])
    else:
        ind_list = ["All"] + _sorted_unique(df["Industry"])
    selected_industry = st.sidebar.selectbox("Industry", ind_list, index=0, key="industry")

    return selected_sector, selected_industry_group, selected_industry


# ---------- Quick filters & search ----------

def render_quick_filters(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with st.sidebar.expander("More filters", expanded=False):
        for col in ["Theme", "Country", "Asset_Type"]:
            if col in df.columns:
                choices = _sorted_unique(df[col])
                sel = st.multiselect(f"{col}", choices, default=[], key=f"qf_{col}")
                out[col] = sel
    return out


def render_search_box() -> str:
    return st.sidebar.text_input("Search Symbol/Name", value="", key="search").strip()


def apply_quick_filters_and_search(df: pd.DataFrame, quick: Dict[str, List[str]], search: str) -> pd.DataFrame:
    out = df.copy()
    for col, sel in quick.items():
        if sel and col in out.columns:
            out = out[out[col].astype(str).isin(sel)]

    if search:
        cols = [c for c in ["Symbol", "Name"] if c in out.columns]
        if cols:
            mask = False
            for c in cols:
                mask = mask | out[c].astype(str).str.contains(search, case=False, na=False)
            out = out[mask]
    return out


# ---------- Column visibility ----------

def column_selector(
    df: pd.DataFrame,
    default_front: Optional[List[str]] = None,
    key_prefix: str = ""
) -> List[str]:
    """
    Let the user select which columns to show.
    key_prefix avoids key collisions when used in multiple places.
    """
    with st.expander("Columns", expanded=False):
        ordered = _frontload_columns(df, default_front or ["Symbol", "Name", "Sector", "Industry Group", "Industry"])
        selected = st.multiselect(
            "Visible columns",
            options=ordered,
            default=ordered,  # show all by default
            key=f"{key_prefix}visible_cols",   # unique key to prevent duplication
        )
    return selected if selected else ordered


# ---------- Summary chips ----------

def show_summary_chips(df: pd.DataFrame) -> None:
    cols = st.columns(3)
    if "Sector" in df.columns:
        cols[0].metric("Sectors", df["Sector"].nunique())
    if "Industry Group" in df.columns:
        cols[1].metric("Industry Groups", df["Industry Group"].nunique())
    if "Industry" in df.columns:
        cols[2].metric("Industries", df["Industry"].nunique())


# ---------- Tables ----------

def show_filtered_table(filtered_df: pd.DataFrame, visible_cols: Optional[List[str]] = None) -> None:
    st.caption("Preview of symbols that match current filters")
    show_summary_chips(filtered_df)

    if visible_cols is None:
        visible_cols = column_selector(filtered_df, key_prefix="preview_")
    table_df = filtered_df[visible_cols] if visible_cols else filtered_df

    col_config = {}
    for c in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[c]):
            col_config[c] = st.column_config.NumberColumn(format="%.3f")
        elif pd.api.types.is_integer_dtype(table_df[c]):
            col_config[c] = st.column_config.NumberColumn(format="%d")

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )


def show_results_table(result_df: pd.DataFrame, visible_cols: Optional[List[str]] = None) -> None:
    st.subheader("Results with Yahoo Finance Metrics")

    default_front = ["Symbol", "Name", "Sector", "Industry Group", "Industry",
                     "Current Price", "PE Ratio", "Market Cap", "Dividend Yield",
                     "52 Week High", "52 Week Low", "Beta", "Volume", "Avg Volume"]
    ordered = _frontload_columns(result_df, default_front)

    if visible_cols is None:
        visible_cols = column_selector(result_df, default_front=default_front, key_prefix="results_")
    table_df = result_df[visible_cols] if visible_cols else result_df[ordered]

    col_config = {}
    for c in table_df.columns:
        if c in {"Current Price", "52 Week High", "52 Week Low"}:
            col_config[c] = st.column_config.NumberColumn(format="%.2f")
        elif c in {"PE Ratio", "Beta"}:
            col_config[c] = st.column_config.NumberColumn(format="%.2f")
        elif c in {"Dividend Yield"}:
            col_config[c] = st.column_config.NumberColumn(format="%.2f%%")
        elif c in {"Market Cap"}:
            col_config[c] = st.column_config.NumberColumn(format="%.0f")
        elif pd.api.types.is_float_dtype(table_df[c]):
            col_config[c] = st.column_config.NumberColumn(format="%.3f")
        elif pd.api.types.is_integer_dtype(table_df[c]):
            col_config[c] = st.column_config.NumberColumn(format="%d")

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )


# ---------- Downloads ----------

def _to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.read()


def download_csv_button(df: pd.DataFrame, filename: str = "results.csv") -> Optional[bytes]:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    excel_bytes = _to_excel_bytes(df)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            key="dl_csv",
        )
    with c2:
        st.download_button(
            label="⬇️ Download Excel (.xlsx)",
            data=excel_bytes,
            file_name=filename.replace(".csv", ".xlsx") if filename.endswith(".csv") else "results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx",
        )

    return csv_bytes
