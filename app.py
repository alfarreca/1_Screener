# app.py — Financial Stock Screener (Pro) + Risk Light v2
# -------------------------------------------------------
# Features:
# - SPX/VIX Risk Light with tri-state: RISK ON / AMBER—PRICE / AMBER—VOL / RISK OFF
# - Daily or Intraday mode (15m/30m/60m), with Refresh button
# - Screener flow using your existing modules: data_loader, analysis, visualization
# - Value/Growth/Momentum + Value-Contrarian score
#
# Requirements: streamlit, pandas, numpy, (yfinance for Risk Light)

import sys
import pathlib
import importlib
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import streamlit as st

# ---------- Optional yfinance for Risk Light ----------
from functools import lru_cache
try:
    import yfinance as yf
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

# ---------- Streamlit page ----------
st.set_page_config(page_title="Financial Stock Screener (Pro)", layout="wide")

# ---------- Ensure repo path ----------
APP_DIR = pathlib.Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# ---------- Local modules ----------
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

# ==================== Risk Light (SPX/VIX) v2 ====================
@lru_cache(maxsize=256)
def _last_price(ticker, period, interval, salt):
    """
    Get last price using yfinance; `salt` is a cache-buster integer.
    Returns float or None.
    """
    if not YF_AVAILABLE:
        return None
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        s = df["Close"].dropna()
        return float(s.iloc[-1]) if not s.empty else None
    except Exception:
        return None


def render_risk_light_sidebar():
    """
    Sidebar controls + compute risk posture.
    Returns a dict with state/metrics.
    """
    st.sidebar.subheader("Risk Light (SPX/VIX)")
    enabled = st.sidebar.checkbox("Show Risk Light", value=True)

    spx_symbol = st.sidebar.text_input("SPX symbol (Yahoo)", value="^GSPC", help="Alt: SPY")
    vix_symbol = st.sidebar.text_input("VIX symbol (Yahoo)", value="^VIX")

    # Data mode: daily or intraday
    mode = st.sidebar.selectbox("Data mode", ["Daily close", "Intraday"])
    if mode == "Daily close":
        period, interval = "7d", "1d"
    else:
        intraday_interval = st.sidebar.selectbox("Intraday interval", ["15m", "30m", "60m"], index=0)
        period, interval = "7d", intraday_interval

    spx_level = st.sidebar.number_input("SPX de-risk level", value=6100.0, step=25.0)
    vix_thr = st.sidebar.number_input("VIX threshold", value=18.0, step=0.5)

    # Refresh button clears cache
    if st.sidebar.button("Refresh risk data"):
        try:
            _last_price.cache_clear()
        except Exception:
            pass

    if not enabled:
        return {
            "enabled": False, "state": "HIDDEN", "reason": "Disabled by user",
            "spx_last": None, "vix_last": None,
            "spx_level": spx_level, "vix_thr": vix_thr,
            "risk_off": None, "amber": None,
            "mode": mode, "interval": interval
        }

    if not YF_AVAILABLE:
        st.sidebar.warning("`yfinance` not installed. Add it to requirements.txt.")

    # Minute salt to auto-refresh roughly each minute on rerun
    salt = int(datetime.now(timezone.utc).strftime("%Y%m%d%H%M"))
    spx_last = _last_price(spx_symbol, period, interval, salt)
    vix_last = _last_price(vix_symbol, period, interval, salt)

    # Tri-state logic
    state, reason, risk_off, amber = "UNKNOWN", "Data unavailable", None, None
    if (spx_last is not None) and (vix_last is not None):
        price_breach = spx_last < spx_level
        vol_breach = vix_last > vix_thr

        if price_breach and vol_breach:
            state, reason, risk_off, amber = "RISK OFF", "Price < level AND Vol > threshold", True, None
        elif price_breach:
            state, reason, risk_off, amber = "AMBER — PRICE", "SPX<{}".format(int(spx_level)), False, "PRICE"
        elif vol_breach:
            state, reason, risk_off, amber = "AMBER — VOL", "VIX>{}".format(int(vix_thr)), False, "VOL"
        else:
            state, reason, risk_off, amber = "RISK ON", "SPX≥{} and VIX≤{}".format(int(spx_level), int(vix_thr)), False, None

    return {
        "enabled": True, "state": state, "reason": reason,
        "spx_last": spx_last, "vix_last": vix_last,
        "spx_level": spx_level, "vix_thr": vix_thr,
        "risk_off": risk_off, "amber": amber,
        "mode": mode, "interval": interval
    }
# ===============================================================


# ---------------- Numeric Filters ----------------
def add_numeric_filters(df: pd.DataFrame):
    """Sidebar numeric filters with safe guards."""
    st.sidebar.subheader("Numeric Filters")
    filters = {}

    def safe_slider(colname, label):
        if colname not in df.columns:
            return
        nums = pd.to_numeric(df[colname], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if nums.empty:
            return
        min_val, max_val = float(nums.min()), float(nums.max())
        if np.isfinite(min_val) and np.isfinite(max_val) and (max_val - min_val) > 0:
            filters[colname] = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))

    safe_slider("PE Ratio", "P/E Ratio")
    safe_slider("Dividend Yield", "Dividend Yield %")
    safe_slider("Market Cap", "Market Cap ($)")

    return filters


def apply_numeric_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply numeric slider filters."""
    out = df.copy()
    for col, (low, high) in filters.items():
        if col in out.columns:
            num_col = pd.to_numeric(out[col], errors="coerce")
            out = out[(num_col >= low) & (num_col <= high)]
    return out


# ---------------- Scoring System ----------------
def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add Value, Growth, Momentum, and Value-Contrarian scores."""
    out = df.copy()

    def as_num(s):
        return pd.to_numeric(s, errors="coerce")

    # Value Score: low P/E + high Dividend Yield
    if "PE Ratio" in out.columns and "Dividend Yield" in out.columns:
        pe = as_num(out["PE Ratio"])
        dy = as_num(out["Dividend Yield"])
        if not pe.dropna().empty and not dy.dropna().empty:
            pe_rank = pe.rank(ascending=True, pct=True)      # lower P/E = better
            dy_rank = dy.rank(ascending=False, pct=True)     # higher yield = better
            out["Value Score"] = ((pe_rank + dy_rank) / 2 * 100).round(1)

    # Growth Score: EPS growth %
    if "EPS Growth %" in out.columns:
        eps = as_num(out["EPS Growth %"])
        if not eps.dropna().empty:
            out["Growth Score"] = (eps.rank(ascending=False, pct=True) * 100).round(1)

    # Momentum Score: 52-week percentile
    if {"Current Price", "52 Week Low", "52 Week High"} <= set(out.columns):
        cp = as_num(out["Current Price"])
        lo = as_num(out["52 Week Low"])
        hi = as_num(out["52 Week High"])
        rng = (hi - lo).replace(0, pd.NA)
        if not rng.dropna().empty:
            out["Momentum Score"] = ((cp - lo) / rng * 100).round(1)

    # Value-Contrarian Score: high value + mid/low momentum (~40% sweet spot)
    if "Value Score" in out.columns and "Momentum Score" in out.columns:
        v = as_num(out["Value Score"]).clip(0, 100) / 100.0
        m = as_num(out["Momentum Score"]).clip(0, 100) / 100.0

        m_target = 0.40    # momentum sweet spot
        falloff = 0.40     # how quickly score fades away from sweet spot
        contrarian_factor = 1.0 - (abs(m - m_target) / falloff)
        contrarian_factor = contrarian_factor.clip(lower=0, upper=1)

        out["Value-Contrarian Score"] = (v * contrarian_factor * 100).round(1)

    return out


# ---------------- Main App ----------------
def main():
    st.title("Financial Stock Screener (Pro)")
    st.caption(
        "Upload an Excel file with Symbol, Sector, Industry Group, Industry (Name optional). "
        "Filter in the sidebar. Fetch Yahoo Finance data for scores and metrics."
    )

    # === Risk Light at top ===
    rl = render_risk_light_sidebar()
    top = st.container()
    with top:
        c1, c2, c3 = st.columns(3)
        c1.metric("S&P 500 (last)", f"{rl['spx_last']:.2f}" if rl["spx_last"] is not None else "—")
        c2.metric("VIX (last)", f"{rl['vix_last']:.2f}" if rl["vix_last"] is not None else "—")
        c3.metric("Tripwire", f"SPX {rl['spx_level']:.0f} / VIX {rl['vix_thr']:.0f}")

        mode_str = f"{rl['mode']} ({rl['interval']})" if rl['mode'] == "Intraday" else rl['mode']
        st.caption(f"Risk Light source: {mode_str}")

        if rl["enabled"]:
            if rl["state"] == "RISK ON":
                st.success(f"✅ {rl['state']} — {rl['reason']}")
            elif str(rl["state"]).startswith("AMBER"):
                st.warning(f"🟡 {rl['state']} — {rl['reason']}")
            elif rl["state"] == "RISK OFF":
                st.error(f"⚠️ {rl['state']} — {rl['reason']}")
            else:
                st.info("ℹ️ Risk Light: data unavailable right now.")

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

    # === Taxonomy filters ===
    selected_sector, selected_industry_group, selected_industry = render_sidebar_filters(df)

    # === Apply taxonomy filters ===
    filtered_df = apply_filters(
        df,
        selected_sector=selected_sector,
        selected_industry_group=selected_industry_group,
        selected_industry=selected_industry,
    )

    # === Quick filters & search ===
    quick = render_quick_filters(filtered_df)
    query = render_search_box()
    filtered_df = apply_quick_filters_and_search(filtered_df, quick, query)

    st.subheader("Filtered Stocks (Before Metrics)")
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
        score_cols = [
            c for c in ["Value Score", "Growth Score", "Momentum Score", "Value-Contrarian Score"]
            if c in result_df.columns
        ]
        if score_cols:
            sort_choice = st.sidebar.selectbox("Sort by Score", ["None"] + score_cols)
            if sort_choice != "None":
                result_df = result_df.sort_values(sort_choice, ascending=False)

        show_results_table(result_df)
        download_csv_button(result_df, "stock_screener_results.csv")


if __name__ == "__main__":
    main()
