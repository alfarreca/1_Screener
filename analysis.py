# analysis.py
from __future__ import annotations

from typing import List, Dict, Any, Iterable
import pandas as pd
import yfinance as yf

# -----------------------------
# Helpers
# -----------------------------

def _as_list(symbols: Iterable[str]) -> List[str]:
    out = []
    for s in symbols or []:
        if s is None:
            continue
        s = str(s).strip().upper()
        if s:
            out.append(s)
    # de-dup but keep order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _first_not_null(*values):
    """Return the first value that is not None/NaN."""
    for v in values:
        try:
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            return v
        except Exception:
            continue
    return None


# -----------------------------
# 1) Taxonomy filtering
# -----------------------------

def apply_filters(
    df: pd.DataFrame,
    selected_sector: str = "All",
    selected_industry_group: str = "All",
    selected_industry: str = "All",
) -> pd.DataFrame:
    """
    Filter by Sector -> Industry Group -> Industry.
    Missing columns are ignored gracefully.
    """
    out = df.copy()

    if "Sector" in out.columns and selected_sector and selected_sector != "All":
        out = out[out["Sector"].astype(str) == selected_sector]

    if "Industry Group" in out.columns and selected_industry_group and selected_industry_group != "All":
        out = out[out["Industry Group"].astype(str) == selected_industry_group]

    if "Industry" in out.columns and selected_industry and selected_industry != "All":
        out = out[out["Industry"].astype(str) == selected_industry]

    return out.reset_index(drop=True)


# -----------------------------
# 2) Yahoo Finance fetch (robust)
# -----------------------------

def fetch_yfinance_data(symbols: Iterable[str]) -> pd.DataFrame:
    """
    Robust Yahoo fetch with:
      - batched yfinance.Tickers() (fewer HTTP calls)
      - fast_info first, get_info() fallback
      - final fallback from .history() to compute price / 52w / volume
    Returns a DataFrame of metrics.
    Failed symbols are stored in df.attrs['failed_symbols'].
    """
    syms = _as_list(symbols)
    if not syms:
        df = pd.DataFrame(columns=[
            "Symbol", "Current Price", "PE Ratio", "Market Cap", "Dividend Yield",
            "52 Week High", "52 Week Low", "Beta", "Volume", "Avg Volume"
        ])
        df.attrs["failed_symbols"] = []
        return df

    tk = yf.Tickers(" ".join(syms))
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for s in syms:
        t = tk.tickers.get(s) or yf.Ticker(s)

        # Prefer fast_info (newer, faster)
        try:
            fi = getattr(t, "fast_info", {}) or {}
            fi_dict = dict(fi) if not isinstance(fi, dict) else fi
        except Exception:
            fi_dict = {}

        # Fallback to slower info
        try:
            info = t.get_info()  # may 404 or be sparsely populated
        except Exception:
            info = {}

        price = _first_not_null(
            fi_dict.get("last_price"),
            fi_dict.get("regular_market_price"),
            info.get("currentPrice"),
            info.get("regularMarketPrice"),
        )
        mcap = _first_not_null(
            fi_dict.get("market_cap"),
            info.get("marketCap"),
        )
        pe = _first_not_null(
            fi_dict.get("trailing_pe"),
            info.get("trailingPE"),
        )
        dy = _first_not_null(
            fi_dict.get("dividend_yield"),
            info.get("dividendYield"),
        )
        hi52 = _first_not_null(
            fi_dict.get("year_high"),
            info.get("fiftyTwoWeekHigh"),
        )
        lo52 = _first_not_null(
            fi_dict.get("year_low"),
            info.get("fiftyTwoWeekLow"),
        )
        beta = _first_not_null(
            info.get("beta"),
            info.get("beta3Year"),
        )
        vol = _first_not_null(
            info.get("volume"),
            info.get("regularMarketVolume"),
        )
        avgvol = _first_not_null(
            fi_dict.get("three_month_average_volume"),
            info.get("averageVolume"),
            info.get("averageDailyVolume3Month"),
        )

        # Final fallback: derive from 1y history if missing
        if any(v is None for v in [price, hi52, lo52, vol, avgvol]):
            try:
                hist = t.history(period="1y", interval="1d", auto_adjust=False)
                if not hist.empty:
                    if price is None:
                        price = float(hist["Close"].iloc[-1])
                    if hi52 is None:
                        hi52 = float(hist["High"].rolling(252, min_periods=1).max().iloc[-1])
                    if lo52 is None:
                        lo52 = float(hist["Low"].rolling(252, min_periods=1).min().iloc[-1])
                    if vol is None:
                        vol = float(hist["Volume"].iloc[-1])
                    if avgvol is None:
                        avgvol = float(hist["Volume"].rolling(63, min_periods=1).mean().iloc[-1])
            except Exception:
                # ignore; we'll mark as failed if still empty
                pass

        # If everything missing, mark failure
        if all(v is None for v in [price, mcap, pe, dy, hi52, lo52, vol, avgvol]):
            failed.append(s)
            continue

        # Normalize dividend yield to percentage if needed
        if dy is not None and isinstance(dy, (int, float)) and dy < 1:
            dy = dy * 100.0

        rows.append(
            {
                "Symbol": s,
                "Current Price": price,
                "PE Ratio": pe,
                "Market Cap": mcap,
                "Dividend Yield": dy,
                "52 Week High": hi52,
                "52 Week Low": lo52,
                "Beta": beta,
                "Volume": vol,
                "Avg Volume": avgvol,
            }
        )

    df = pd.DataFrame(rows)

    # Ensure numeric dtypes where possible
    for col in [
        "Current Price", "PE Ratio", "Market Cap", "Dividend Yield",
        "52 Week High", "52 Week Low", "Beta", "Volume", "Avg Volume"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.attrs["failed_symbols"] = failed
    return df


# -----------------------------
# 3) Merge back to original rows
# -----------------------------

def merge_results(filtered_df: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge metrics onto filtered_df by Symbol.
    Keeps the original row order.
    """
    if financial_data is None or financial_data.empty:
        # return filtered_df with metric columns missing
        return filtered_df.copy()

    base = filtered_df.copy()
    # normalize Symbol to string/upper for join
    if "Symbol" in base.columns:
        base["Symbol"] = base["Symbol"].astype(str).str.strip().str.upper()
    if "Symbol" in financial_data.columns:
        financial_data = financial_data.copy()
        financial_data["Symbol"] = financial_data["Symbol"].astype(str).str.strip().str.upper()

    out = base.merge(financial_data, on="Symbol", how="left", sort=False)
    return out


# -----------------------------
# 4) Convenience wrapper (optional)
# -----------------------------

def fetch_yahoo_metrics(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper used by some versions of app.py:
    - extracts symbols from filtered_df
    - fetches Yahoo metrics
    - merges them back
    """
    if filtered_df is None or filtered_df.empty or "Symbol" not in filtered_df.columns:
        return filtered_df.copy()

    symbols = filtered_df["Symbol"].tolist()
    metrics = fetch_yfinance_data(symbols)
    return merge_results(filtered_df, metrics)
