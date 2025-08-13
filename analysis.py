# analysis.py
from __future__ import annotations
from typing import List, Dict, Any, Iterable
import pandas as pd
import yfinance as yf

# ---------------- helpers ----------------

def _as_list(symbols: Iterable[str]) -> List[str]:
    out = []
    for s in symbols or []:
        if s is None:
            continue
        s = str(s).strip().upper()
        if s:
            out.append(s)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def _first_not_null(*values):
    for v in values:
        try:
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
        except Exception:
            continue
    return None

# crude map from industry -> industry group (extend as you wish)
_GICS_GROUP_MAP: Dict[str, str] = {
    "Semiconductors": "Semiconductors & Semiconductor Equipment",
    "Semiconductor": "Semiconductors & Semiconductor Equipment",
    "Software": "Software & Services",
    "Information Technology Services": "Software & Services",
    "IT Services": "Software & Services",
    "Gold": "Metals & Mining",
    "Oil & Gas": "Energy",
    "Oil & Gas E&P": "Energy",
    "Oil & Gas Refining & Marketing": "Energy",
    "Banks": "Banks",
    "Biotechnology": "Pharmaceuticals, Biotechnology & Life Sciences",
    "Pharmaceuticals": "Pharmaceuticals, Biotechnology & Life Sciences",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Aerospace & Defense": "Capital Goods",
    "Automobile Manufacturers": "Automobiles & Components",
    "Auto Components": "Automobiles & Components",
    "Retailing": "Retailing",
    "Consumer Services": "Consumer Services",
}

# very light theme inference using keywords in industry/summary
_THEME_RULES: List[tuple[str, str]] = [
    ("gold", "Gold"),
    ("silver", "Silver"),
    ("copper", "Copper"),
    ("lithium", "Lithium / EV Materials"),
    ("semiconductor", "Semiconductors"),
    ("chip", "Semiconductors"),
    ("ai ", "AI"),
    (" artificial intelligence", "AI"),
    ("cloud", "Cloud"),
    ("saas", "SaaS"),
    ("gaming", "Gaming"),
    ("renewable", "Renewables"),
    ("solar", "Renewables - Solar"),
    ("wind", "Renewables - Wind"),
    ("ev ", "Electric Vehicles"),
    ("electric vehicle", "Electric Vehicles"),
    ("cybersecurity", "Cybersecurity"),
    ("oil", "Oil & Gas"),
    ("gas", "Oil & Gas"),
    ("uranium", "Uranium"),
    ("fintech", "Fintech"),
]

def _infer_industry_group(industry: str | None) -> str | None:
    if not industry:
        return None
    txt = industry.lower()
    for key, group in _GICS_GROUP_MAP.items():
        if key.lower() in txt:
            return group
    # fallback: title-case industry as group
    return industry

def _infer_theme(sector: str | None, industry: str | None, summary: str | None) -> str | None:
    text = " ".join([sector or "", industry or "", (summary or "")[:2000]]).lower()
    for needle, theme in _THEME_RULES:
        if needle in text:
            return theme
    return None

# ---------------- taxonomy-aware fetch ----------------

def fetch_yfinance_data(symbols: Iterable[str]) -> pd.DataFrame:
    """
    Fetch metrics + taxonomy from Yahoo.
    - Sector & Industry from info
    - Industry Group inferred from Industry
    - Theme inferred by keywords (best-effort)
    Returns df; failures stored in df.attrs['failed_symbols'].
    """
    syms = _as_list(symbols)
    if not syms:
        df = pd.DataFrame(columns=[
            "Symbol", "Current Price", "PE Ratio", "Market Cap", "Dividend Yield",
            "52 Week High", "52 Week Low", "Beta", "Volume", "Avg Volume",
            "Sector", "Industry Group", "Industry", "Theme"
        ])
        df.attrs["failed_symbols"] = []
        return df

    tk = yf.Tickers(" ".join(syms))
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for s in syms:
        t = tk.tickers.get(s) or yf.Ticker(s)

        try:
            fi = dict(getattr(t, "fast_info", {}) or {})
        except Exception:
            fi = {}
        try:
            info = t.get_info()
        except Exception:
            info = {}

        # taxonomy
        sector = _first_not_null(info.get("sector"))
        industry = _first_not_null(info.get("industry"))
        long_summary = _first_not_null(info.get("longBusinessSummary"))

        industry_group = _infer_industry_group(industry)
        theme = _infer_theme(sector, industry, long_summary)

        # metrics
        price = _first_not_null(fi.get("last_price"), fi.get("regular_market_price"),
                                info.get("currentPrice"), info.get("regularMarketPrice"))
        mcap = _first_not_null(fi.get("market_cap"), info.get("marketCap"))
        pe   = _first_not_null(fi.get("trailing_pe"), info.get("trailingPE"))
        dy   = _first_not_null(fi.get("dividend_yield"), info.get("dividendYield"))
        hi52 = _first_not_null(fi.get("year_high"), info.get("fiftyTwoWeekHigh"))
        lo52 = _first_not_null(fi.get("year_low"),  info.get("fiftyTwoWeekLow"))
        beta = _first_not_null(info.get("beta"), info.get("beta3Year"))
        vol  = _first_not_null(info.get("volume"), info.get("regularMarketVolume"))
        avgv = _first_not_null(fi.get("three_month_average_volume"),
                               info.get("averageVolume"),
                               info.get("averageDailyVolume3Month"))

        # fallback from history if needed
        if any(v is None for v in [price, hi52, lo52, vol, avgv]):
            try:
                hist = t.history(period="1y", interval="1d", auto_adjust=False)
                if not hist.empty:
                    price = price if price is not None else float(hist["Close"].iloc[-1])
                    hi52  = hi52  if hi52  is not None else float(hist["High"].rolling(252, min_periods=1).max().iloc[-1])
                    lo52  = lo52  if lo52  is not None else float(hist["Low"].rolling(252, min_periods=1).min().iloc[-1])
                    vol   = vol   if vol   is not None else float(hist["Volume"].iloc[-1])
                    avgv  = avgv  if avgv  is not None else float(hist["Volume"].rolling(63, min_periods=1).mean().iloc[-1])
            except Exception:
                pass

        if all(v is None for v in [price, mcap, pe, dy, hi52, lo52, vol, avgv, sector, industry]):
            failed.append(s)
            continue

        if isinstance(dy, (int, float)) and dy is not None and dy < 1:
            dy = dy * 100.0

        rows.append({
            "Symbol": s,
            "Current Price": price,
            "PE Ratio": pe,
            "Market Cap": mcap,
            "Dividend Yield": dy,
            "52 Week High": hi52,
            "52 Week Low": lo52,
            "Beta": beta,
            "Volume": vol,
            "Avg Volume": avgv,
            "Sector": sector,
            "Industry": industry,
            "Industry Group": industry_group,
            "Theme": theme,
        })

    df = pd.DataFrame(rows)

    # enforce numeric types where possible
    for col in ["Current Price","PE Ratio","Market Cap","Dividend Yield",
                "52 Week High","52 Week Low","Beta","Volume","Avg Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.attrs["failed_symbols"] = failed
    return df

# ---------------- merging ----------------

def merge_results(filtered_df: pd.DataFrame, financial_data: pd.DataFrame, *, override_taxonomy: bool = False) -> pd.DataFrame:
    """
    Merge metrics & taxonomy onto filtered_df by Symbol.
    If override_taxonomy=True, replace Sector/Industry/Industry Group/Theme
    with Yahoo-derived values even when upload already had them.
    Otherwise, only fill where missing.
    """
    if financial_data is None or financial_data.empty:
        return filtered_df.copy()

    base = filtered_df.copy()
    if "Symbol" in base.columns:
        base["Symbol"] = base["Symbol"].astype(str).str.strip().str.upper()
    fd = financial_data.copy()
    if "Symbol" in fd.columns:
        fd["Symbol"] = fd["Symbol"].astype(str).str.strip().str.upper()

    out = base.merge(fd, on="Symbol", how="left", suffixes=("", "_y"), sort=False)

    for col in ["Sector", "Industry Group", "Industry", "Theme"]:
        y = f"{col}_y"
        if y in out.columns:
            if override_taxonomy or col not in out.columns:
                out[col] = out[y]
            else:
                out[col] = out[col].where(out[col].notna() & (out[col].astype(str).str.strip() != ""), out[y])
            out.drop(columns=[y], inplace=True, errors="ignore")

    return out

# convenience wrapper if some app versions use it
def fetch_yahoo_metrics(filtered_df: pd.DataFrame, override_taxonomy: bool = False) -> pd.DataFrame:
    if filtered_df is None or filtered_df.empty or "Symbol" not in filtered_df.columns:
        return filtered_df.copy()
    symbols = filtered_df["Symbol"].tolist()
    metrics = fetch_yfinance_data(symbols)
    return merge_results(filtered_df, metrics, override_taxonomy=override_taxonomy)
