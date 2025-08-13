# app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import data_loader
import analysis
import visualization


# ---------------- Helpers ----------------
def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# ---------------- Risk Light (SPX / VIX) ----------------
@st.cache_data(show_spinner=False, ttl=300)
def _last_price(symbol: str, mode: str) -> tuple[float | None, str]:
    """
    Return last price for a Yahoo symbol and a short 'source' note.
    mode: 'Daily close' or 'Recent (approx)'
    """
    try:
        tk = yf.Ticker(symbol)
        if mode == "Daily close":
            # last available daily close
            hist = tk.history(period="7d", interval="1d", auto_adjust=False, raise_errors=False)
            if hist is not None and not hist.empty and "Close" in hist:
                return float(hist["Close"].dropna().iloc[-1]), "Daily close"
        else:
            # recent price snapshot: try fast_info, otherwise 1d/1m last
            fi = getattr(tk, "fast_info", None)
            if fi:
                try:
                    return float(fi["last_price"]), "Fast info"
                except Exception:
                    pass
            hist = tk.history(period="5d", interval="5m", auto_adjust=False, raise_errors=False)
            if hist is not None and not hist.empty and "Close" in hist:
                return float(hist["Close"].dropna().iloc[-1]), "Recent 5m"
    except Exception:
        pass
    return None, "N/A"


def render_risk_light() -> tuple[bool, dict]:
    st.sidebar.markdown("### Risk Light (SPX/VIX)")
    show = st.sidebar.checkbox("Show Risk Light", value=True, key="risk_onoff")
    spx_symbol = st.sidebar.text_input("SPX symbol (Yahoo)", value="^GSPC", help="S&P 500 index symbol on Yahoo")
    vix_symbol = st.sidebar.text_input("VIX symbol (Yahoo)", value="^VIX", help="CBOE VIX index symbol on Yahoo")
    mode = st.sidebar.selectbox("Data mode", ["Daily close", "Recent (approx)"], index=0)
    spx_level = st.sidebar.number_input("SPX de-risk level", value=6100.0, step=25.0)
    vix_th = st.sidebar.number_input("VIX threshold", value=18.0, step=0.5)
    refresh = st.sidebar.button("Refresh risk data")

    # Compute prices (cache bust by adding dummy arg on refresh)
    price_spx, src_spx = _last_price(spx_symbol, mode if not refresh else mode + " ")
    price_vix, src_vix = _last_price(vix_symbol, mode if not refresh else mode + " ")

    info = {
        "spx_symbol": spx_symbol,
        "vix_symbol": vix_symbol,
        "mode": mode,
        "spx_level": spx_level,
        "vix_th": vix_th,
        "price_spx": price_spx,
        "price_vix": price_vix,
        "source_note": src_spx if src_spx == src_vix else f"{src_spx} / {src_vix}",
    }
    return show, info


def show_risk_banner(info: dict):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("S&P 500 (last)", f"{info['price_spx']:.2f}" if info["price_spx"] is not None else "—")
    with c2:
        st.metric("VIX (last)", f"{info['price_vix']:.2f}" if info["price_vix"] is not None else "—")
    with c3:
        st.metric("Tripwire", f"SPX {int(info['spx_level'])} / VIX {int(info['vix_th'])}")

    st.caption(f"Risk Light source: {info['source_note']}")

    ok_spx = info["price_spx"] is not None and info["price_spx"] >= info["spx_level"]
    ok_vix = info["price_vix"] is not None and info["price_vix"] <= info["vix_th"]
    is_on = ok_spx and ok_vix

    if is_on:
        st.success(f"✅ RISK ON — SPX≥{int(info['spx_level'])} and VIX≤{int(info['vix_th'])}")
    else:
        # Show which leg is failing
        msg_parts = []
        if info["price_spx"] is None:
            msg_parts.append("SPX n/a")
        elif not ok_spx:
            msg_parts.append(f"SPX<{int(info['spx_level'])}")
        if info["price_vix"] is None:
            msg_parts.append("VIX n/a")
        elif not ok_vix:
            msg_parts.append(f"VIX>{int(info['vix_th'])}")
        st.warning("⚠️ RISK OFF — " + " & ".join(msg_parts))


# ---------------- Sidebar: Numeric Filters ----------------
def add_numeric_filters(df: pd.DataFrame) -> dict:
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
    out = df.copy()

    # Value Score: low P/E + high Dividend Yield
    if {"PE Ratio", "Dividend Yield"} <= set(out.columns):
        pe = as_num(out["PE Ratio"])
        dy = as_num(out["Dividend Yield"])
        if pe.notna().any() and dy.notna().any():
            pe_rank = pe.rank(ascending=True, pct=True)
            dy_rank = dy.rank(ascending=False, pct=True)
            out["Value Score"] = ((pe_rank + dy_rank) / 2 * 100).round(1)

    # Growth Score if provided later (EPS Growth %)
    if "EPS Growth %" in out.columns:
        eps = as_num(out["EPS Growth %"])
        if eps.notna().any():
            out["Growth Score"] = (eps.rank(ascending=False, pct=True) * 100).round(1)

    # Momentum Score: 52-week percentile (float-safe)
    if {"Current Price", "52 Week Low", "52 Week High"} <= set(out.columns):
        cp = as_num(out["Current Price"])
        lo = as_num(out["52 Week Low"])
        hi = as_num(out["52 Week High"])
        rng = (hi - lo)
        rng = rng.replace(0, np.nan)
        momentum = (cp - lo) / rng * 100
        out["Momentum Score"] = momentum.astype(float).round(1)

    # Value-Contrarian Score: value × preference for ~40% momentum
    if {"Value Score", "Momentum Score"} <= set(out.columns):
        v = as_num(out["Value Score"]).clip(0, 100) / 100.0
        m = as_num(out["Momentum Score"]).clip(0, 100) / 100.0
        m_target, falloff = 0.40, 0.40
        contrarian = 1.0 - (np.abs(m - m_target) / falloff)
        contrarian = np.clip(contrarian, 0.0, 1.0)
        out["Value-Contrarian Score"] = (v * contrarian * 100).round(1)

    return out


# ---------------- App ----------------
def main():
    st.set_page_config(page_title="Financial Stock Screener (Pro)", layout="wide")
    st.title("💹 Financial Stock Screener (Pro)")
    st.caption(
        "Upload an Excel/CSV with columns: **Symbol, Name, Country, Asset_Type, Notes**. "
        "We immediately enrich Sector/Industry for filtering, then fetch Yahoo metrics for scores."
    )

    # Risk Light (optional, top-of-page)
    show, risk_info = render_risk_light()
    if show:
        show_risk_banner(risk_info)
        st.write("")  # small spacer

    # Upload
    uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
    if not uploaded:
        st.info("Upload a file to begin.")
        return

    # Load + validate with auto header repair
    df = data_loader.load_data(uploaded)
    if df is None or df.empty:
        st.error("Could not read the file or it was empty.")
        return
    if not data_loader.validate_columns(df):
        st.error(f"Your file must include these columns: {sorted(data_loader.REQUIRED_COLUMNS)}")
        return

    visualization.checkbox_show_raw(df)

    # === Taxonomy-first enrichment (BEFORE filters)
    with st.spinner("Loading Sector / Industry from Yahoo…"):
        symbols_for_tax = (
            df["Symbol"].astype(str).str.strip().str.upper().replace("", np.nan).dropna().unique().tolist()
            if "Symbol" in df.columns else []
        )
        if symbols_for_tax:
            tax_df = analysis.fetch_taxonomy_only(symbols_for_tax)
            # override so filters are available immediately
            df = analysis.merge_results(df, tax_df, override_taxonomy=True)

    # Filters (safe even if some symbols lack taxonomy)
    sel_sector, sel_ig, sel_ind = visualization.render_sidebar_filters(df)

    # Apply taxonomy filters server-side
    filtered_df = analysis.apply_filters(
        df,
        selected_sector=sel_sector,
        selected_industry_group=sel_ig,
        selected_industry=sel_ind,
    )

    # Quick filters + search
    quick = visualization.render_quick_filters(filtered_df)
    query = visualization.render_search_box()
    filtered_df = visualization.apply_quick_filters_and_search(filtered_df, quick, query)

    # Preview (pre-metrics)
    st.subheader("Filtered Stocks (before metrics)")
    st.caption(f"{len(filtered_df)} rows match your filters.")
    if filtered_df.empty:
        st.warning("No rows match your filters. Adjust and try again.")
        return
    visualization.show_filtered_table(filtered_df)

    # === Full metrics fetch + scoring
    if st.button("Fetch Financial Data from Yahoo Finance"):
        with st.spinner("Fetching data from Yahoo…"):
            symbols2 = (
                filtered_df["Symbol"]
                .astype(str).str.strip().str.upper()
                .replace("", np.nan).dropna().unique().tolist()
            )
            if not symbols2:
                st.warning("No valid symbols found.")
                return
            fin = analysis.fetch_yfinance_data(symbols2)

        failed = getattr(fin, "attrs", {}).get("failed_symbols", [])
        if failed:
            st.warning(f"Could not fetch data for: {', '.join(failed)}")

        result_df = analysis.merge_results(filtered_df, fin, override_taxonomy=False)
        result_df = add_scores(result_df)

        # Sliders on result metrics
        sliders = add_numeric_filters(result_df)
        result_df = apply_numeric_filters(result_df, sliders)

        # Sort by score
        score_cols = [c for c in ["Value Score", "Growth Score", "Momentum Score", "Value-Contrarian Score"]
                      if c in result_df.columns]
        if score_cols:
            choice = st.sidebar.selectbox("Sort by Score", ["None"] + score_cols, index=0)
            if choice != "None":
                result_df = result_df.sort_values(choice, ascending=False, na_position="last")

        visualization.show_results_table(result_df)
        visualization.download_csv_button(result_df, filename="screener_results.csv")


if __name__ == "__main__":
    main()
