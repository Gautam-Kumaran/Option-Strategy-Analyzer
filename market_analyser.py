"""
Module 2 — Market Condition Analyser
NSE Options Decision Tool

Reads:
    data/{SYMBOL}_options.csv
    data/{SYMBOL}_history.csv

Outputs a dict (and prints as JSON) with:
    iv_rank, iv_regime, atm_iv, spot,
    trend, adx, sma20, sma50,
    max_call_oi_strike, max_put_oi_strike, near_resistance, near_support, expiry,
    pcr, pcr_signal, symbol

Usage:
    python market_analyser.py HDFCBANK
    python market_analyser.py HDFCBANK --date 2024-11-29   # analyse a specific past date
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load options CSV and history CSV for a symbol."""
    opts_path = Path(f"data/{symbol}_options.csv")
    hist_path = Path(f"data/{symbol}_history.csv")

    if not opts_path.exists():
        raise FileNotFoundError(
            f"{opts_path} not found. Run: python data_puller.py --stock {symbol}"
        )
    if not hist_path.exists():
        raise FileNotFoundError(
            f"{hist_path} not found. Run: python data_puller.py --stock {symbol}"
        )

    opts = pd.read_csv(opts_path, parse_dates=["date", "expiry"])
    hist = pd.read_csv(hist_path, parse_dates=["Date"], index_col="Date")
    hist.sort_index(inplace=True)

    return opts, hist


def nearest_strike(strikes: np.ndarray, spot: float) -> float:
    """Return the strike closest to spot price."""
    return strikes[np.argmin(np.abs(strikes - spot))]


# ── IV Rank ───────────────────────────────────────────────────────────────────

def compute_iv_rank(opts: pd.DataFrame, as_of_date=None) -> dict:
    """
    IV Rank = (current ATM IV − 52wk low IV) / (52wk high IV − 52wk low IV) × 100

    Steps:
    1. Find ATM strike on the most recent (or as_of) date
    2. Average call + put IV at that strike → current ATM IV
    3. Repeat for every date in the dataset → daily ATM IV series
    4. Compute rank within that series
    """
    if as_of_date is None:
        as_of_date = opts["date"].max()

    # Drop rows with missing IV (--no-iv run or computation failures)
    opts_iv = opts.dropna(subset=["iv"])
    if opts_iv.empty:
        raise ValueError(
            "No IV data found. Re-run data_puller.py without --no-iv flag."
        )

    # ── current ATM IV ──
    today_rows = opts_iv[opts_iv["date"] == as_of_date]
    if today_rows.empty:
        raise ValueError(f"No option data for date {as_of_date.date()}")

    spot = today_rows["underlying"].iloc[0]
    strikes_today = today_rows["strike"].unique()
    atm = nearest_strike(strikes_today, spot)

    atm_today = today_rows[today_rows["strike"] == atm]
    atm_iv = atm_today["iv"].mean()  # avg of CE + PE at ATM

    # ── daily ATM IV series ──
    def _daily_atm_iv(grp):
        s = grp["underlying"].iloc[0]
        atm_strike = nearest_strike(grp["strike"].unique(), s)
        return grp[grp["strike"] == atm_strike]["iv"].mean()

    daily_iv = (
        opts_iv
        .groupby("date")[["strike", "underlying", "iv"]]
        .apply(_daily_atm_iv)
        .dropna()
    )

    iv_52_high = daily_iv.max()
    iv_52_low  = daily_iv.min()

    if iv_52_high == iv_52_low:
        iv_rank = 50.0  # degenerate case — treat as neutral
    else:
        iv_rank = (atm_iv - iv_52_low) / (iv_52_high - iv_52_low) * 100

    return {
        "iv_rank":    round(float(iv_rank), 1),
        "iv_regime":  "High" if iv_rank >= 70 else ("Low" if iv_rank <= 30 else "Neutral"),
        "atm_iv":     round(float(atm_iv), 4),
        "spot":       round(float(spot), 2),
    }


# ── Trend Detection ───────────────────────────────────────────────────────────

def compute_trend(hist: pd.DataFrame, as_of_date=None) -> dict:
    """
    Trend = SMA crossover (direction) + ADX (strength)

    Rules:
    - ADX < 20  → Range-bound (overrides SMA)
    - ADX ≥ 25  → Trending; use SMA for direction
    - 20 ≤ ADX < 25 → weak trend; still use SMA but flag
    - SMA20 > SMA50 → Bullish
    - SMA20 < SMA50 → Bearish
    """
    if as_of_date is not None:
        hist = hist[hist.index <= as_of_date]

    if len(hist) < 50:
        raise ValueError(
            f"Need at least 50 days of history for SMA50. Got {len(hist)}."
        )

    close = hist["Close"]
    high  = hist["High"]
    low   = hist["Low"]

    # ── SMAs ──
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    # ── ADX (Wilder's smoothing, period=14) ──
    period = 14

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move)   & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move)   & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=hist.index)
    minus_dm_s = pd.Series(minus_dm, index=hist.index)

    # Wilder's smoothing = EWM with alpha = 1/period
    alpha = 1 / period
    atr       = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di   = 100 * plus_dm_s.ewm(alpha=alpha,  adjust=False).mean() / atr
    minus_di  = 100 * minus_dm_s.ewm(alpha=alpha, adjust=False).mean() / atr

    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    # ── latest values ──
    adx_val  = float(adx.iloc[-1])
    sma20_val = float(sma20.iloc[-1])
    sma50_val = float(sma50.iloc[-1])

    if adx_val < 20:
        trend = "Range-bound"
    elif sma20_val > sma50_val:
        trend = "Bullish"
    else:
        trend = "Bearish"

    return {
        "trend": trend,
        "adx":   round(adx_val,   2),
        "sma20": round(sma20_val, 2),
        "sma50": round(sma50_val, 2),
    }


# ── OI Analysis ───────────────────────────────────────────────────────────────

def compute_oi_analysis(opts: pd.DataFrame, as_of_date=None) -> dict:
    """
    Find max Call OI strike (resistance) and max Put OI strike (support)
    on the nearest expiry as of the given date.
    """
    if as_of_date is None:
        as_of_date = opts["date"].max()

    today_rows = opts[opts["date"] == as_of_date]
    if today_rows.empty:
        raise ValueError(f"No option data for date {as_of_date.date()}")

    nearest_expiry = today_rows["expiry"].min()
    chain = today_rows[today_rows["expiry"] == nearest_expiry]

    calls = chain[chain["option_type"] == "CE"].groupby("strike")["oi"].sum()
    puts  = chain[chain["option_type"] == "PE"].groupby("strike")["oi"].sum()

    if calls.empty or puts.empty:
        raise ValueError("Insufficient OI data — missing calls or puts in chain.")

    max_call_strike = int(calls.idxmax())
    max_put_strike  = int(puts.idxmax())

    spot = float(chain["underlying"].iloc[0])

    # "Near" = within 2% of spot
    near_resistance = abs(spot - max_call_strike) / spot < 0.02
    near_support    = abs(spot - max_put_strike)  / spot < 0.02

    support_broken    = spot < max_put_strike
    resistance_broken = spot > max_call_strike

    return {
        "max_call_oi_strike": max_call_strike,   # implied resistance
        "max_put_oi_strike":  max_put_strike,    # implied support
        "near_resistance":    bool(near_resistance),
        "near_support":       bool(near_support),
        "expiry":             str(nearest_expiry.date()),
        "support_broken":     bool(support_broken),
        "resistance_broken":  bool(resistance_broken),
    }


# ── PCR ───────────────────────────────────────────────────────────────────────

def compute_pcr(opts: pd.DataFrame, as_of_date=None) -> dict:
    """
    PCR = total Put OI / total Call OI on nearest expiry.

    > 1.2 → Bearish sentiment
    < 0.8 → Bullish sentiment
    else  → Neutral
    """
    if as_of_date is None:
        as_of_date = opts["date"].max()

    today_rows = opts[opts["date"] == as_of_date]
    nearest_expiry = today_rows["expiry"].min()
    chain = today_rows[today_rows["expiry"] == nearest_expiry]

    total_call_oi = chain[chain["option_type"] == "CE"]["oi"].sum()
    total_put_oi  = chain[chain["option_type"] == "PE"]["oi"].sum()

    if total_call_oi == 0:
        raise ValueError("Total call OI is zero — cannot compute PCR.")

    pcr = total_put_oi / total_call_oi

    if pcr > 1.2:
        signal = "Bullish"
    elif pcr < 0.8:
        signal = "Bearish"
    else:
        signal = "Neutral"

    # Derive clean symbol name from the data
    raw_symbol = chain["symbol"].iloc[0] if "symbol" in chain.columns else "UNKNOWN"

    return {
        "pcr":        round(float(pcr), 3),
        "pcr_signal": signal,
        "symbol":     str(raw_symbol),
    }

def compute_direction(result: dict) -> dict:
    """
    Combine trend, PCR, and OI wall signals into a single directional score.
    
    Scoring:
      +1 for each Bullish signal, -1 for each Bearish signal
    
    Signals used:
      - SMA trend (Bullish / Bearish / Range-bound)
      - PCR signal (Bullish / Bearish / Neutral)
      - support_broken → Bearish
      - resistance_broken → Bullish
    
    Output:
      score ≥ +2  → Bullish
      score ≤ -2  → Bearish
      score = 0   → Neutral
      |score| = 1 → Conflict (signals disagree, not strong enough either way)
    """
    score = 0

    if result["trend"] == "Bullish":
        score += 1
    elif result["trend"] == "Bearish":
        score -= 1

    if result["pcr_signal"] == "Bullish":
        score += 1
    elif result["pcr_signal"] == "Bearish":
        score -= 1

    if result.get("resistance_broken"):
        score += 1
    if result.get("support_broken"):
        score -= 1

    if score >= 2:
        direction = "Bullish"
    elif score <= -2:
        direction = "Bearish"
    elif score == 0:
        direction = "Neutral"
    else:
        direction = "Conflict"

    return {
        "direction":       direction,
        "direction_score": score,
    }
# ── Main Analyser ─────────────────────────────────────────────────────────────

def analyse(symbol: str, as_of_date=None) -> dict:
    """
    Run all 4 analyses and return a single merged dict.

    Args:
        symbol:      Stock ticker, e.g. "HDFCBANK"
        as_of_date:  Optional date string "YYYY-MM-DD". Defaults to latest date in data.
    """
    opts, hist = load_data(symbol)

    # Normalise as_of_date
    if as_of_date is not None:
        as_of_date = pd.Timestamp(as_of_date)
        # Snap to nearest available trading date on or before as_of_date
        available_dates = opts["date"].unique()
        available_dates.sort()
        valid = available_dates[available_dates <= as_of_date]
        if len(valid) == 0:
            raise ValueError(f"No data on or before {as_of_date.date()}")
        as_of_date = pd.Timestamp(valid[-1])
        print(f"[info] Analysing as of {as_of_date.date()}", file=sys.stderr)
    else:
        as_of_date = opts["date"].max()
        print(f"[info] Analysing as of {as_of_date.date()} (latest available)", file=sys.stderr)

    result = {}
    result.update(compute_iv_rank(opts, as_of_date))
    result.update(compute_trend(hist, as_of_date))
    result.update(compute_oi_analysis(opts, as_of_date))
    result.update(compute_pcr(opts, as_of_date))
    result["as_of_date"] = str(as_of_date.date())

    # Compute directional signal
    result.update(compute_direction(result))

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _pretty_print(result: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 52)
    print(f"  Market Analysis — {result['symbol']}  ({result['as_of_date']})")
    print("=" * 52)

    print(f"\n  Spot Price : ₹{result['spot']:,.2f}")
    print(f"  Expiry     : {result['expiry']}")

    print(f"\n  IV Rank    : {result['iv_rank']}%  [{result['iv_regime']} IV]")
    print(f"  ATM IV     : {result['atm_iv'] * 100:.1f}%")

    print(f"\n  Trend      : {result['trend']}")
    print(f"  ADX        : {result['adx']}")
    print(f"  SMA 20/50  : {result['sma20']} / {result['sma50']}")

    print(f"\n  Resistance : ₹{result['max_call_oi_strike']:,}  (max call OI)"
          + ("  ← spot is near here!" if result["near_resistance"] else ""))
    print(f"  Support    : ₹{result['max_put_oi_strike']:,}  (max put OI)"
          + ("  ← spot is near here!" if result["near_support"] else ""))

    print(f"\n  PCR        : {result['pcr']}  [{result['pcr_signal']} sentiment]")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module 2 — Market Condition Analyser"
    )
    parser.add_argument("stock", help="Stock symbol, e.g. HDFCBANK")
    parser.add_argument(
        "--date", default=None,
        help="Analyse as of this date (YYYY-MM-DD). Defaults to latest in data."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print raw JSON output instead of formatted summary"
    )
    args = parser.parse_args()

    try:
        result = analyse(args.stock.upper(), as_of_date=args.date)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        _pretty_print(result)
        print("[tip] Pass --json to get raw JSON for piping into Module 3.")
