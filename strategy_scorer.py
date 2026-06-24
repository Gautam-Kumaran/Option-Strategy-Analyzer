"""
Module 3 — Strategy Scorer
Reads market analysis output (JSON) and recommends top 2 option strategies.

Usage:
    python market_analyser.py HDFCBANK --json | python strategy_scorer.py
    python strategy_scorer.py HDFCBANK
    python strategy_scorer.py HDFCBANK --json
"""

import sys
import json
import argparse
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, date


# ─────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────

def _d1_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_call_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(S - K, 0)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if d1 is None:
        return max(K - S, 0)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def prob_above(S, K, T, r, sigma):
    """Probability spot finishes above K at expiry (risk-neutral)."""
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if d2 is None:
        return 1.0 if S > K else 0.0
    return norm.cdf(d2)


def prob_below(S, K, T, r, sigma):
    return 1.0 - prob_above(S, K, T, r, sigma)


def days_to_expiry(expiry_str):
    """Return time to expiry in years."""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    today = date.today()
    dte = (exp - today).days
    return max(dte, 1) / 365.0


# ─────────────────────────────────────────────
# Chain loading helpers
# ─────────────────────────────────────────────

def load_chain(symbol, expiry, as_of_date=None):
    """
    Load options chain for a given symbol and expiry from the saved CSV.
    Returns DataFrame with columns: strike, option_type, last_price, iv, volume, oi
    Filters to the most recent date available (or as_of_date if provided).
    """
    path = f"data/{symbol}_options.csv"
    df = pd.read_csv(path, parse_dates=["date", "expiry"])

    # Normalise expiry format
    exp_dt = pd.to_datetime(expiry)
    df = df[df["expiry"] == exp_dt]

    if as_of_date:
        target = pd.to_datetime(as_of_date)
    else:
        target = df["date"].max()

    df = df[df["date"] == target].copy()

    # Use last_price; fall back to close if missing
    if "last_price" not in df.columns:
        df["last_price"] = df["close"]
    else:
        df["last_price"] = df["last_price"].fillna(df["close"])

    df = df[["strike", "option_type", "last_price", "iv", "volume", "oi"]].copy()
    df["option_type"] = df["option_type"].str.upper().str.strip()
    return df


def get_atm_strike(chain, spot):
    """Return the strike closest to spot."""
    strikes = chain["strike"].unique()
    return min(strikes, key=lambda x: abs(x - spot))


def nth_otm_strike(chain, spot, option_type, n=1):
    """
    Return the Nth OTM strike for a given option type.
    For CE: Nth strike above ATM.
    For PE: Nth strike below ATM.
    n=1 → first OTM, n=2 → second OTM, etc.
    """
    option_type = option_type.upper()
    atm = get_atm_strike(chain, spot)
    strikes = sorted(chain["strike"].unique())

    if option_type == "CE":
        otm = [s for s in strikes if s > atm]
        return otm[n - 1] if len(otm) >= n else otm[-1]
    else:
        otm = [s for s in strikes if s < atm][::-1]
        return otm[n - 1] if len(otm) >= n else otm[-1]


def get_premium(chain, strike, option_type):
    """Get last_price for a given strike and option type. Returns 0 if not found."""
    option_type = option_type.upper()
    row = chain[(chain["strike"] == strike) & (chain["option_type"] == option_type)]
    if row.empty:
        return 0.0
    return float(row["last_price"].iloc[0])


def get_iv(chain, strike, option_type):
    """Get IV for a given strike and option type."""
    option_type = option_type.upper()
    row = chain[(chain["strike"] == strike) & (chain["option_type"] == option_type)]
    if row.empty or row["iv"].isna().all():
        return 0.25  # fallback
    return float(row["iv"].iloc[0])


# ─────────────────────────────────────────────
# Strategy implementations
# ─────────────────────────────────────────────

R = 0.065  # risk-free rate (approx RBI repo)


def long_straddle(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])
    atm = get_atm_strike(chain, S)

    call_prem = get_premium(chain, atm, "CE")
    put_prem = get_premium(chain, atm, "PE")
    iv = (get_iv(chain, atm, "CE") + get_iv(chain, atm, "PE")) / 2

    net_debit = call_prem + put_prem
    be_upper = atm + net_debit
    be_lower = atm - net_debit
    max_loss = net_debit
    max_profit = "Unlimited"

    # PoP = prob spot moves beyond either breakeven
    pop = prob_above(S, be_upper, T, R, iv) + prob_below(S, be_lower, T, R, iv)

    return {
        "strategy": "Long Straddle",
        "type": "debit",
        "legs": [
            {"action": "BUY", "type": "CE", "strike": atm, "premium": call_prem},
            {"action": "BUY", "type": "PE", "strike": atm, "premium": put_prem},
        ],
        "net_premium": round(net_debit, 2),
        "max_profit": max_profit,
        "max_loss": round(max_loss, 2),
        "be_upper": round(be_upper, 2),
        "be_lower": round(be_lower, 2),
        "prob_of_profit": round(pop * 100, 1),
    }


def long_strangle(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])

    otm_ce_strike = nth_otm_strike(chain, S, "CE", n=1)
    otm_pe_strike = nth_otm_strike(chain, S, "PE", n=1)

    call_prem = get_premium(chain, otm_ce_strike, "CE")
    put_prem = get_premium(chain, otm_pe_strike, "PE")
    iv = (get_iv(chain, otm_ce_strike, "CE") + get_iv(chain, otm_pe_strike, "PE")) / 2

    net_debit = call_prem + put_prem
    be_upper = otm_ce_strike + net_debit
    be_lower = otm_pe_strike - net_debit
    max_loss = net_debit
    max_profit = "Unlimited"

    pop = prob_above(S, be_upper, T, R, iv) + prob_below(S, be_lower, T, R, iv)

    return {
        "strategy": "Long Strangle",
        "type": "debit",
        "legs": [
            {"action": "BUY", "type": "CE", "strike": otm_ce_strike, "premium": call_prem},
            {"action": "BUY", "type": "PE", "strike": otm_pe_strike, "premium": put_prem},
        ],
        "net_premium": round(net_debit, 2),
        "max_profit": max_profit,
        "max_loss": round(max_loss, 2),
        "be_upper": round(be_upper, 2),
        "be_lower": round(be_lower, 2),
        "prob_of_profit": round(pop * 100, 1),
    }


def short_strangle(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])

    # Use OI walls if available, else 1st OTM
    ce_strike = market.get("max_call_oi_strike")
    pe_strike = market.get("max_put_oi_strike")

    if ce_strike is None or ce_strike <= S * 1.02:
        ce_strike = nth_otm_strike(chain, S, "CE", n=2)

    if pe_strike is None or pe_strike >= S * 0.98:
        pe_strike = nth_otm_strike(chain, S, "PE", n=2)

    call_prem = get_premium(chain, ce_strike, "CE")
    put_prem = get_premium(chain, pe_strike, "PE")
    iv = (get_iv(chain, ce_strike, "CE") + get_iv(chain, pe_strike, "PE")) / 2

    net_credit = call_prem + put_prem
    be_upper = ce_strike + net_credit
    be_lower = pe_strike - net_credit
    max_profit = net_credit
    max_loss = "Unlimited"

    pop = prob_below(S, be_upper, T, R, iv) - prob_below(S, be_lower, T, R, iv)

    return {
        "strategy": "Short Strangle",
        "type": "credit",
        "legs": [
            {"action": "SELL", "type": "CE", "strike": ce_strike, "premium": call_prem},
            {"action": "SELL", "type": "PE", "strike": pe_strike, "premium": put_prem},
        ],
        "net_premium": round(net_credit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": max_loss,
        "be_upper": round(be_upper, 2),
        "be_lower": round(be_lower, 2),
        "prob_of_profit": round(pop * 100, 1),
    }


def iron_condor(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])

    # Sell strikes at OI walls, buy one strike further out
    ce_strike = market.get("max_call_oi_strike")
    pe_strike = market.get("max_put_oi_strike")

    if ce_strike is None or ce_strike <= S * 1.02:
        ce_strike = nth_otm_strike(chain, S, "CE", n=2)

    if pe_strike is None or pe_strike >= S * 0.98:
        pe_strike = nth_otm_strike(chain, S, "PE", n=2)

    sell_ce = ce_strike
    sell_pe = pe_strike
    
    buy_ce = nth_otm_strike(chain, S, "CE", n=1)
    buy_pe = nth_otm_strike(chain, S, "PE", n=1)

    sell_ce_prem = get_premium(chain, sell_ce, "CE")
    buy_ce_prem = get_premium(chain, buy_ce, "CE")
    sell_pe_prem = get_premium(chain, sell_pe, "PE")
    buy_pe_prem = get_premium(chain, buy_pe, "PE")

    net_credit = (sell_ce_prem - buy_ce_prem) + (sell_pe_prem - buy_pe_prem)
    call_width = buy_ce - sell_ce
    put_width = sell_pe - buy_pe
    max_loss = max(call_width, put_width) - net_credit
    be_upper = sell_ce + net_credit
    be_lower = sell_pe - net_credit

    iv = (get_iv(chain, sell_ce, "CE") + get_iv(chain, sell_pe, "PE")) / 2
    pop = prob_below(S, be_upper, T, R, iv) - prob_below(S, be_lower, T, R, iv)

    return {
        "strategy": "Iron Condor",
        "type": "credit",
        "legs": [
            {"action": "SELL", "type": "CE", "strike": sell_ce, "premium": sell_ce_prem},
            {"action": "BUY",  "type": "CE", "strike": buy_ce,  "premium": buy_ce_prem},
            {"action": "SELL", "type": "PE", "strike": sell_pe, "premium": sell_pe_prem},
            {"action": "BUY",  "type": "PE", "strike": buy_pe,  "premium": buy_pe_prem},
        ],
        "net_premium": round(net_credit, 2),
        "max_profit": round(net_credit, 2),
        "max_loss": round(max_loss, 2),
        "be_upper": round(be_upper, 2),
        "be_lower": round(be_lower, 2),
        "prob_of_profit": round(pop * 100, 1),
    }


def bull_call_spread(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])
    atm = get_atm_strike(chain, S)
    otm_ce = nth_otm_strike(chain, S, "CE", n=1)

    buy_prem = get_premium(chain, atm, "CE")
    sell_prem = get_premium(chain, otm_ce, "CE")
    iv = get_iv(chain, atm, "CE")

    net_debit = buy_prem - sell_prem
    max_profit = (otm_ce - atm) - net_debit
    max_loss = net_debit
    be = atm + net_debit

    pop = prob_above(S, be, T, R, iv)

    return {
        "strategy": "Bull Call Spread",
        "type": "debit",
        "legs": [
            {"action": "BUY",  "type": "CE", "strike": atm,    "premium": buy_prem},
            {"action": "SELL", "type": "CE", "strike": otm_ce, "premium": sell_prem},
        ],
        "net_premium": round(net_debit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "be_upper": round(be, 2),
        "be_lower": None,
        "prob_of_profit": round(pop * 100, 1),
    }


def bear_put_spread(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])
    atm = get_atm_strike(chain, S)
    otm_pe = nth_otm_strike(chain, S, "PE", n=1)

    buy_prem = get_premium(chain, atm, "PE")
    sell_prem = get_premium(chain, otm_pe, "PE")
    iv = get_iv(chain, atm, "PE")

    net_debit = buy_prem - sell_prem
    max_profit = (atm - otm_pe) - net_debit
    max_loss = net_debit
    be = atm - net_debit

    pop = prob_below(S, be, T, R, iv)

    return {
        "strategy": "Bear Put Spread",
        "type": "debit",
        "legs": [
            {"action": "BUY",  "type": "PE", "strike": atm,    "premium": buy_prem},
            {"action": "SELL", "type": "PE", "strike": otm_pe, "premium": sell_prem},
        ],
        "net_premium": round(net_debit, 2),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "be_upper": None,
        "be_lower": round(be, 2),
        "prob_of_profit": round(pop * 100, 1),
    }


def bull_put_spread(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])

    sell_pe = nth_otm_strike(chain, S, "PE", n=1)
    buy_pe = nth_otm_strike(chain, S, "PE", n=2)

    sell_prem = get_premium(chain, sell_pe, "PE")
    buy_prem = get_premium(chain, buy_pe, "PE")
    iv = get_iv(chain, sell_pe, "PE")

    net_credit = sell_prem - buy_prem
    max_profit = net_credit
    max_loss = (sell_pe - buy_pe) - net_credit
    be = sell_pe - net_credit

    pop = prob_above(S, be, T, R, iv)

    return {
        "strategy": "Bull Put Spread",
        "type": "credit",
        "legs": [
            {"action": "SELL", "type": "PE", "strike": sell_pe, "premium": sell_prem},
            {"action": "BUY",  "type": "PE", "strike": buy_pe,  "premium": buy_prem},
        ],
        "net_premium": round(net_credit, 2),
        "max_profit": round(net_credit, 2),
        "max_loss": round(max_loss, 2),
        "be_upper": None,
        "be_lower": round(be, 2),
        "prob_of_profit": round(pop * 100, 1),
    }


def bear_call_spread(market, chain):
    S = market["spot"]
    T = days_to_expiry(market["expiry"])

    sell_ce = nth_otm_strike(chain, S, "CE", n=1)
    buy_ce = nth_otm_strike(chain, S, "CE", n=2)

    sell_prem = get_premium(chain, sell_ce, "CE")
    buy_prem = get_premium(chain, buy_ce, "CE")
    iv = get_iv(chain, sell_ce, "CE")

    net_credit = sell_prem - buy_prem
    max_profit = net_credit
    max_loss = (buy_ce - sell_ce) - net_credit
    be = sell_ce + net_credit

    pop = prob_below(S, be, T, R, iv)

    return {
        "strategy": "Bear Call Spread",
        "type": "credit",
        "legs": [
            {"action": "SELL", "type": "CE", "strike": sell_ce, "premium": sell_prem},
            {"action": "BUY",  "type": "CE", "strike": buy_ce,  "premium": buy_prem},
        ],
        "net_premium": round(net_credit, 2),
        "max_profit": round(net_credit, 2),
        "max_loss": round(max_loss, 2),
        "be_upper": round(be, 2),
        "be_lower": None,
        "prob_of_profit": round(pop * 100, 1),
    }


# ─────────────────────────────────────────────
# Strategy selection matrix
# ─────────────────────────────────────────────

def select_strategies(market):
    iv = market["iv_regime"]        # "Low", "Neutral", "High"
    direction = market["direction"]  # "Bullish", "Bearish", "Neutral", "Conflict"
    adx = market["adx"]

    # Conflict → no trade regardless of IV
    if direction == "Conflict":
        return []

    if iv == "High":
        if direction in ("Neutral", "Range-bound"):
            return [iron_condor, short_strangle]
        elif direction == "Bullish":
            return [bull_put_spread, iron_condor]
        elif direction == "Bearish":
            return [bear_call_spread, iron_condor]

    elif iv == "Low":
        # Only recommend straddle/strangle if there's a breakout setup
        has_breakout = adx >= 18 and direction in ("Bullish", "Bearish")
        if has_breakout:
            return [long_straddle, long_strangle]
        elif direction == "Bullish":
            return [bull_call_spread, bull_put_spread]
        elif direction == "Bearish":
            return [bear_put_spread, bear_call_spread]
        else:
            return []  # Low IV, Neutral direction → no trade

    elif iv == "Neutral":
        if direction == "Bullish":
            return [bull_call_spread, bull_put_spread]
        elif direction == "Bearish":
            return [bear_put_spread, bear_call_spread]
        else:
            return []  # Neutral IV + Neutral direction → no trade

    return []  # fallback

# ─────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────

def pretty_print(market, results):
    SEP = "=" * 52
    sep = "-" * 52

    print(f"\n{SEP}")
    print(f"  Strategy Recommendations — {market['symbol']} ({market['as_of_date']})")
    print(SEP)
    print(f"  Spot: ₹{market['spot']:.2f}  |  Expiry: {market['expiry']}")
    print(f"  IV Rank: {market['iv_rank']:.1f}% [{market['iv_regime']} IV]  |  "
          f"Trend: {market['trend']}")
    print(sep)

    if isinstance(results, dict) and results.get("no_trade"):
        print(f"\n  ⚠  NO TRADE")
        print(f"  {results['reason']}")
        print(f"\n{SEP}\n")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  Recommendation {i}: {r['strategy'].upper()}  [{r['type'].upper()}]")
        print(sep)

        print("  Legs:")
        for leg in r["legs"]:
            sign = "+" if leg["action"] == "BUY" else "-"
            print(f"    {sign} {leg['action']} {leg['type']} ₹{leg['strike']}  "
                  f"@ ₹{leg['premium']:.2f}")

        print()
        net_label = "Net Debit" if r["type"] == "debit" else "Net Credit"
        print(f"  {net_label}      : ₹{r['net_premium']:.2f} per share  "
              f"(₹{r['net_premium'] * 550:.0f} per lot)")

        if r["max_profit"] == "Unlimited":
            print(f"  Max Profit      : Unlimited")
        else:
            print(f"  Max Profit      : ₹{r['max_profit']:.2f} per share  "
                  f"(₹{r['max_profit'] * 550:.0f} per lot)")

        if r["max_loss"] == "Unlimited":
            print(f"  Max Loss        : Unlimited")
        else:
            print(f"  Max Loss        : ₹{r['max_loss']:.2f} per share  "
                  f"(₹{r['max_loss'] * 550:.0f} per lot)")

        if r["be_upper"] and r["be_lower"]:
            print(f"  Breakevens      : ₹{r['be_lower']:.2f} ↓  /  ₹{r['be_upper']:.2f} ↑")
        elif r["be_upper"]:
            print(f"  Breakeven       : ₹{r['be_upper']:.2f} ↑")
        elif r["be_lower"]:
            print(f"  Breakeven       : ₹{r['be_lower']:.2f} ↓")

        print(f"  Prob of Profit  : {r['prob_of_profit']}%")

    print(f"\n{SEP}")
    print("  [tip] Use --json to get raw JSON output.")
    print(f"{SEP}\n")


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
def _no_trade_reason(market):
    iv = market["iv_regime"]
    direction = market["direction"]
    adx = market["adx"]

    if direction == "Conflict":
        return (
            f"Signals conflict (score={market['direction_score']}): "
            f"trend={market['trend']}, PCR={market['pcr_signal']}, "
            f"support_broken={market.get('support_broken')}, "
            f"resistance_broken={market.get('resistance_broken')}. "
            f"No edge — sit out."
        )
    if iv == "Low" and adx < 18:
        return (
            f"Low IV (rank={market['iv_rank']}%) but no momentum "
            f"(ADX={adx}). Straddle/strangle won't pay off. Sit out."
        )
    if iv in ("Low", "Neutral") and direction == "Neutral":
        return (
            f"{iv} IV and no directional edge. "
            f"Nothing to trade — sit out."
        )
    return "No suitable strategy found for current market conditions."

def score(market):
    chain = load_chain(market["symbol"], market["expiry"], market.get("as_of_date"))
    strategy_fns = select_strategies(market)
    if not strategy_fns:
        reason = _no_trade_reason(market)
        return {"no_trade": True, "reason": reason}

    results = [fn(market, chain) for fn in strategy_fns]
    return results


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Options Strategy Scorer")
    parser.add_argument("symbol", nargs="?", help="Stock symbol e.g. HDFCBANK")
    parser.add_argument("--date", help="Analysis date YYYY-MM-DD (default: latest)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    # Read market data — either from stdin (piped from Module 2) or run Module 2 inline
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        market = json.loads(raw)
    elif args.symbol:
        # Import and run market_analyser directly
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "market_analyser",
            os.path.join(os.path.dirname(__file__), "market_analyser.py")
        )
        ma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ma)
        market = ma.analyse(args.symbol, as_of_date=args.date)
    else:
        print("Error: provide a symbol or pipe JSON from market_analyser.py")
        print("  python market_analyser.py HDFCBANK --json | python strategy_scorer.py")
        print("  python strategy_scorer.py HDFCBANK")
        sys.exit(1)

    results = score(market)

    if args.json:
        output = {"market": market, "recommendations": results}
        print(json.dumps(output, indent=2, default=str))
    else:
        pretty_print(market, results)


if __name__ == "__main__":
    main()
