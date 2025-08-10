import io
import base64
from typing import Tuple, Optional, List, Dict

import pandas as pd
import streamlit as st
import yfinance as yf

# ---------- Small helpers ----------

def _sorted_unique(series: pd.Series) -> List[str]:
    return sorted(series.dropna().astype(str).unique().tolist())

def _frontload_columns(df: pd.DataFrame, front: List[str]) -> List[str]:
    front = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return front + rest

def _human_mc(x) -> str:
    """Humanize market cap numbers."""
    try:
        v = float(x)
    except Exception:
        return ""
    sign = "-" if v < 0 else ""
    v = abs(v)
    for unit in ["", "K", "M", "B", "T"]:
        if v < 1000:
            return f"{sign}{v:.0f}{unit}"
        v /= 1000
    return f"{sign}{v:.0f}T"

def _as_0_100(col: pd.Series) -> pd.Series:
    """Clamp to [0,100] for score bars."""
    x = pd.to_numeric(col, errors="coerce")
    return x.clip(lower=0, upper=100).round(1)

@st.cache_data(show_spinner=False)
def _sparkline_png_bytes(symbol: str, period: str = "1mo") -> Optional[bytes]:
    """
    Return PNG bytes for a tiny sparkline (or None on failure).
    Cached per (symbol, period) to stay fast on Streamlit Cloud.
    """
    try:
        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if data.empty or "Close" not in data:
            return None

        # Lazy headless matplotlib import so module import won't fail if matplotlib missing
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(2, 0.5))
        ax.plot(data["Close"], linewidth=1)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# ---------- Raw data toggle ----------

def checkbox_show_raw(df: pd.DataFrame) -> None:
    if st.checkbox("Show raw uploaded data"):
        with st.expander("Raw data", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

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
    with st.expander("Columns", expanded=False):
        ordered = _frontload_columns(df, default_front or ["Symbol", "Name", "Sector", "Industry Group", "Industry"])
        selected = st.multiselect(
            "Visible columns",
            options=ordered,
            default=ordered,
            key=f"{key_prefix}visible_cols",
        )
    return selected if selected else ordered

# ---------- Summary & display options ----------

def show_summary_chips(df: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Rows", len(df))
    if "Sector" in df.columns:
        cols[1].metric("Sectors", df["Sector"].nunique())
    if "Industry Group" in df.columns:
        cols[2].metric("Industry Groups", df["Industry Group"].nunique())
    if "Industry" in df.columns:
        cols[3].metric("Industries", df["Industry"].nunique())

def display_options() -> dict:
    with st.expander("Display options", expanded=False):
        compact = st.checkbox("Compact mode", value=True, key="compact_mode")
    return {"compact": compact}

# ---------- Tables ----------

def show_filtered_table(filtered_df: pd.DataFrame, visible_cols: Optional[List[str]] = None) -> None:
    st.caption("Preview of symbols that match current filters")
    show_summary_chips(filtered_df)
    opts = display_options()

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
        height=360 if opts["compact"] else None,
    )

def show_results_table(result_df: pd.DataFrame, visible_cols: Optional[List[str]] = None) -> None:
    st.subheader("Results with Yahoo Finance Metrics")

    # Normalize scores for progress bars
    for score_col in ["Value Score", "Growth Score", "Momentum Score", "Value-Contrarian Score"]:
        if score_col in result_df.columns:
            result_df[score_col] = _as_0_100(result_df[score_col])

    # Humanized Market Cap column
    if "Market Cap" in result_df.columns:
        result_df = result_df.copy()
        result_df["Market Cap (fmt)"] = result_df["Market Cap"].apply(_human_mc)

    # Sparkline bytes column (fast thanks to cache)
    if "Symbol" in result_df.columns:
        result_df["Chart"] = result_df["Symbol"].apply(_sparkline_png_bytes)

    default_front = [
        "Symbol", "Name", "Chart", "Sector", "Industry Group", "Industry",
        "Current Price", "PE Ratio", "Market Cap (fmt)", "Dividend Yield",
        "52 Week High", "52 Week Low", "Beta", "Volume", "Avg Volume",
        "Value Score", "Growth Score", "Momentum Score", "Value-Contrarian Score"
    ]
    ordered = _frontload_columns(result_df, default_front)

    if visible_cols is None:
        visible_cols = column_selector(result_df, default_front=default_front, key_prefix="results_")
    table_df = result_df[visible_cols] if visible_cols else result_df[ordered]

    # Column config: scores as progress bars; chart as ImageColumn
    col_config = {}
    for sc in ["Value Score", "Growth Score", "Momentum Score", "Value-Contrarian Score"]:
        if sc in table_df.columns:
            col_config[sc] = st.column_config.ProgressColumn(sc, min_value=0, max_value=100, format="%.0f")

    if "Chart" in table_df.columns:
        col_config["Chart"] = st.column_config.ImageColumn(
            "Chart",
            help="30-day price sparkline",
            width="small",
        )

    for c in table_df.columns:
        if c in {"Current Price", "52 Week High", "52 Week Low"}:
            col_config[c] = st.column_config.NumberColumn(format="%.2f")
        elif c in {"PE Ratio", "Beta"}:
            col_config[c] = st.column_config.NumberColumn(format="%.2f")
        elif c in {"Dividend Yield"}:
            col_config[c] = st.column_config.NumberColumn(format="%.2f%%")
        elif c in {"Volume", "Avg Volume"}:
            col_config[c] = st.column_config.NumberColumn(format="%.0f")
        elif c == "Market Cap":
            col_config[c] = st.column_config.NumberColumn(format="%.0f")
        # Market Cap (fmt) is text already

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
    try:
        excel_bytes = _to_excel_bytes(df)
    except Exception:
        excel_bytes = None

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
        if excel_bytes is not None:
            st.download_button(
                label="⬇️ Download Excel (.xlsx)",
                data=excel_bytes,
                file_name=filename.replace(".csv", ".xlsx") if filename.endswith(".csv") else "results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_xlsx",
            )
    return csv_bytes
