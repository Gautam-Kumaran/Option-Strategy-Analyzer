"""
Module 1: Data Puller
=====================
Downloads F&O bhavcopy data for a date range using nselib,
filters for the requested stock, and saves clean CSVs.

Data sources:
  - nselib derivatives.fno_bhav_copy()  → daily F&O data (options + futures)
  - yfinance                            → 1-year daily OHLCV (spot price history)

Usage
-----
  python data_puller.py --stock HDFCBANK
  python data_puller.py --stock RELIANCE --from 2024-06-01 --to 2024-11-30
  python data_puller.py --stock NIFTY --no-iv

Output (in data/)
-----------------
  {SYMBOL}_options.csv  — all options rows for the stock across the date range
  {SYMBOL}_history.csv  — daily OHLCV from yfinance

Install
-------
  pip install nselib yfinance pandas numpy scipy tqdm
"""

import argparse
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm
from tqdm import tqdm

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.065   # ~6.5% India 91-day T-bill

# Columns we keep from the raw bhavcopy (34 columns → 10 we care about)
# Raw name          → clean name
COL_MAP = {
    "TradDt":           "date",
    "TckrSymb":         "symbol",
    "XpryDt":           "expiry",
    "OptnTp":           "option_type",   # CE or PE
    "StrkPric":         "strike",
    "OpnPric":          "open",
    "HghPric":          "high",
    "LwPric":           "low",
    "ClsPric":          "close",         # EOD settlement price
    "LastPric":         "last_price",    # last traded price (can differ from close)
    "TtlTradgVol":      "volume",        # number of contracts traded
    "OpnIntrst":        "oi",            # open interest
    "ChngInOpnIntrst":  "oi_change",
    "UndrlygPric":      "underlying",    # spot price at that time (from NSE)
}


# ─────────────────────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────────────────────

def trading_days(start: date, end: date) -> list:
    """All weekdays between start and end inclusive."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


# ─────────────────────────────────────────────────────────────────────────────
# nselib downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_one_day(d: date) -> pd.DataFrame | None:
    """
    Download F&O bhavcopy for one trading day via nselib.
    Returns filtered+renamed DataFrame or None if the day is a holiday.

    nselib handles the NSE session internally — no cookie juggling needed.
    nselib date format: 'DD-MM-YYYY'
    """
    from nselib import derivatives
    date_str = d.strftime("%d-%m-%Y")
    try:
        df = derivatives.fno_bhav_copy(trade_date=date_str)
        if df is None or df.empty:
            return None
        # Keep only options (exclude futures whose OptnTp is NaN)
        if "OptnTp" in df.columns:
            df = df[df["OptnTp"].isin(["CE", "PE"])]
        # Rename to clean column names
        rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)
        keep = [v for v in COL_MAP.values() if v in df.columns]
        return df[keep].copy()
    except Exception:
        return None   # holiday / NSE down for that day


def download_bhavcopy_range(symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Download daily F&O bhavcopy for every trading day in [start, end],
    filter for the given symbol, and return a combined DataFrame.

    We do this sequentially (not parallel) to be polite to NSE servers.
    With a small sleep between requests it takes ~2-3 minutes for 1 year.
    """
    days = trading_days(start, end)
    print(f"  Fetching F&O bhavcopy: {len(days)} trading days ({start} → {end})")
    print(f"  Filtering for: {symbol}\n")

    results = []
    skipped = 0

    for d in tqdm(days, unit="day", ncols=70):
        df = download_one_day(d)
        if df is None:
            skipped += 1
            continue
        # Filter to our symbol
        if "symbol" in df.columns:
            filtered = df[df["symbol"].str.strip() == symbol.upper()]
            if not filtered.empty:
                results.append(filtered)
        time.sleep(0.3)   # polite delay — NSE rate-limits aggressive scrapers

    if skipped > 0:
        print(f"\n  ℹ {skipped} days skipped (holidays / no data)")

    if not results:
        raise ValueError(
            f"No F&O data found for '{symbol}' in the date range.\n"
            "Check:\n"
            "  1. Symbol is correct (e.g. HDFCBANK, RELIANCE, NIFTY)\n"
            "  2. The stock has F&O contracts on NSE\n"
            "  3. Internet connection is stable"
        )

    combined = pd.concat(results, ignore_index=True)
    combined = _clean_options_df(combined)

    print(f"\n  ✓ {len(combined):,} option records | "
          f"{combined['date'].nunique()} trading days")
    print(f"  ✓ Strike range: "
          f"₹{combined['strike'].min():,.0f} – ₹{combined['strike'].max():,.0f}")
    expiries = sorted(combined["expiry"].dropna().unique())
    print(f"  ✓ Expiries: {[str(e.date()) for e in expiries[:4]]} ...")

    return combined


def _clean_options_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, cast numerics, drop junk rows."""
    df = df.copy()

    for col in ["date", "expiry"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["strike", "open", "high", "low", "close", "last_price",
                "volume", "oi", "oi_change", "underlying"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical fields
    df = df.dropna(subset=["date", "expiry", "strike", "close"])

    # Drop zero-close rows (contract not traded that day)
    df = df[df["close"] > 0]

    df["option_type"] = df["option_type"].str.strip().str.upper()
    df = df[df["option_type"].isin(["CE", "PE"])]

    df = df.sort_values(["date", "expiry", "strike", "option_type"])
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes & IV
# ─────────────────────────────────────────────────────────────────────────────
#
# Finance concept:
# Bhavcopy gives us the closing price of each option contract.
# We reverse-engineer IV from that price using Black-Scholes.
# This gives us "historical IV" — what the market was pricing on each
# past date. Module 2 uses this to compute IV Rank.
#
# IV Rank = (today's IV − 52wk low IV) / (52wk high IV − 52wk low IV)
# This tells us whether IV is currently cheap or expensive vs its own history.

def bs_price(S, K, T, r, sigma, option_type="CE"):
    """Black-Scholes theoretical option price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "CE" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_iv(market_price, S, K, T, r=RISK_FREE_RATE, option_type="CE"):
    """Back out IV using Brent's method. Returns np.nan if it can't converge."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    intrinsic = max(0.0, (S - K) if option_type == "CE" else (K - S))
    if market_price < intrinsic * 0.95:
        return np.nan
    try:
        iv = brentq(
            lambda sig: bs_price(S, K, T, r, sig, option_type) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=200
        )
        return iv if 0.01 <= iv <= 5.0 else np.nan
    except (ValueError, RuntimeError):
        return np.nan


def add_iv_column(options_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IV for every row using spot price from history and time to expiry.
    The bhavcopy also has UndrlygPric (spot at that time) — we use that when
    available, falling back to the daily close from yfinance.
    """
    print("  Computing implied volatility...")
    print("  (Takes ~60s for 1 year of data — runs once, saved to CSV)")

    # Build date → close price lookup from yfinance history
    yf_spot = {
        pd.Timestamp(k).date(): float(v)
        for k, v in history_df["Close"].to_dict().items()
    }

    def _iv(row):
        d   = row["date"].date()
        exp = row["expiry"].date()
        T   = max((exp - d).days, 1) / 365.0

        # Prefer the underlying price NSE gives us in the bhavcopy
        S = row.get("underlying", np.nan)
        if pd.isna(S) or S <= 0:
            S = yf_spot.get(d, np.nan)
        if pd.isna(S) or S <= 0:
            return np.nan

        return compute_iv(row["close"], S, row["strike"], T,
                          option_type=row["option_type"])

    options_df = options_df.copy()
    options_df["iv"] = options_df.apply(_iv, axis=1)

    valid_pct = options_df["iv"].notna().mean() * 100
    print(f"  ✓ IV computed — valid for {valid_pct:.0f}% of rows")
    return options_df


# ─────────────────────────────────────────────────────────────────────────────
# yfinance spot + history
# ─────────────────────────────────────────────────────────────────────────────

INDEX_MAP = {
    "NIFTY":     "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY":  "NIFTY_FIN_SERVICE.NS",
}

def fetch_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance for the full date range."""
    yf_sym = INDEX_MAP.get(symbol.upper(), f"{symbol.upper()}.NS")
    print(f"  Fetching price history ({yf_sym})...")
    ticker  = yf.Ticker(yf_sym)
    history = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True,
    )
    if history.empty:
        raise ValueError(
            f"yfinance returned no data for '{yf_sym}'.\n"
            "Check the symbol is a valid NSE ticker."
        )
    history = history[["Open", "High", "Low", "Close", "Volume"]].copy()
    history.index = pd.to_datetime(history.index).tz_localize(None)
    print(f"  ✓ History: {len(history)} days | "
          f"₹{history['Close'].iloc[0]:,.2f} → ₹{history['Close'].iloc[-1]:,.2f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Summary & save
# ─────────────────────────────────────────────────────────────────────────────

def display_summary(symbol: str, options: pd.DataFrame, history: pd.DataFrame):
    """Print a quick summary of what was pulled."""
    spot       = history["Close"].iloc[-1]
    latest_dt  = options["date"].max()
    latest     = options[options["date"] == latest_dt]
    near_exp   = latest["expiry"].min()
    chain      = latest[latest["expiry"] == near_exp]
    calls      = chain[chain["option_type"] == "CE"]
    puts       = chain[chain["option_type"] == "PE"]

    atm_iv = np.nan
    pcr    = np.nan

    if not calls.empty:
        atm_idx    = (calls["strike"] - spot).abs().idxmin()
        atm_strike = calls.loc[atm_idx, "strike"]
        c_iv = calls.loc[atm_idx, "iv"] if "iv" in calls.columns else np.nan
        p_iv_row = puts[puts["strike"] == atm_strike]
        p_iv = p_iv_row["iv"].values[0] if (not p_iv_row.empty and "iv" in puts.columns) else np.nan
        atm_iv = np.nanmean([c_iv, p_iv])

    call_oi = calls["oi"].sum()
    put_oi  = puts["oi"].sum()
    if call_oi > 0:
        pcr = put_oi / call_oi

    print(f"\n{'═'*58}")
    print(f"  SUMMARY — {symbol}  (latest: {latest_dt.date()})")
    print(f"{'═'*58}")
    print(f"  Spot price       : ₹{spot:>10,.2f}")
    print(f"  Nearest expiry   : {near_exp.date()}")
    print(f"  ATM IV           : {atm_iv:.1%}" if not np.isnan(atm_iv) else "  ATM IV           : —")
    print(f"  Put-Call Ratio   : {pcr:.3f}"    if not np.isnan(pcr)    else "  Put-Call Ratio   : —")
    print(f"  Total call OI    : {call_oi:>12,.0f}")
    print(f"  Total put  OI    : {put_oi:>12,.0f}")
    print(f"{'─'*58}")
    print(f"  Options records  : {len(options):,}")
    print(f"  Trading days     : {options['date'].nunique()}")
    print(f"  History days     : {len(history)}")
    print(f"{'═'*58}")


def save_csvs(symbol: str, options: pd.DataFrame,
              history: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    opt_path  = os.path.join(output_dir, f"{symbol}_options.csv")
    hist_path = os.path.join(output_dir, f"{symbol}_history.csv")
    options.to_csv(opt_path,  index=False)
    history.to_csv(hist_path, index=True)
    print(f"\n  Saved → {opt_path}  ({os.path.getsize(opt_path)/1e6:.1f} MB)")
    print(f"  Saved → {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def pull_data(symbol: str, start: date = None, end: date = None,
              do_iv: bool = True, output_dir: str = "data",
              save: bool = True) -> dict:
    """
    Full pipeline: download → clean → IV → save.

    Returns dict: {symbol, options (DataFrame), history (DataFrame), start, end}
    """
    symbol = symbol.upper().strip()
    end    = end   or date.today() - timedelta(days=1)
    start  = start or end - timedelta(days=365)

    print(f"\n{'═'*58}")
    print(f"  NSE Options Data Puller — {symbol}")
    print(f"  {start} → {end}")
    print(f"{'═'*58}\n")

    options = download_bhavcopy_range(symbol, start, end)
    history = fetch_history(symbol, start, end)

    if do_iv:
        options = add_iv_column(options, history)

    display_summary(symbol, options, history)

    if save:
        save_csvs(symbol, options, history, output_dir)

    print(f"\n  ✓ Module 1 complete. Ready for Module 2.\n")
    return {"symbol": symbol, "options": options,
            "history": history, "start": start, "end": end}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Options Data Puller — Module 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_puller.py --stock HDFCBANK
  python data_puller.py --stock RELIANCE --from 2024-06-01 --to 2024-11-30
  python data_puller.py --stock NIFTY --no-iv
  python data_puller.py --stock INFY --from 2024-11-01 --to 2024-11-30 --no-save
        """
    )
    parser.add_argument("--stock",   required=True)
    parser.add_argument("--from",    dest="start", default=None,
                        help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("--to",      dest="end",   default=None,
                        help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--no-iv",   action="store_true",
                        help="Skip IV computation (faster)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--output",  default="data")
    args = parser.parse_args()

    def _d(s):
        try:    return datetime.strptime(s, "%Y-%m-%d").date()
        except: print(f"Bad date: {s}"); sys.exit(1)

    try:
        pull_data(
            symbol=args.stock,
            start=_d(args.start) if args.start else None,
            end=_d(args.end)     if args.end   else None,
            do_iv=not args.no_iv,
            output_dir=args.output,
            save=not args.no_save,
        )
    except ValueError as e:
        print(f"\n  Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(0)
