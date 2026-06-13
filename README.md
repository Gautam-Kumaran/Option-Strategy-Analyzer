# NSE Option Strategy Analyzer

This project is a Python tool for analysing NSE stock and index options and suggesting possible option strategies.

It combines historical option data, implied volatility, trend indicators, open interest and put-call ratio.

> This project is for learning and research only. It is not financial advice.

## Project Structure

```text
data_puller.py
market_analyser.py
strategy_scorer.py
data/
```

## Data Puller

`data_puller.py` downloads:

- NSE F&O bhavcopy data using `nselib`
- Historical price data using `yfinance`
- Option prices, volume and open interest
- Implied volatility calculated using Black-Scholes

Example:

```bash
python data_puller.py --stock HDFCBANK
```

You can also use a custom date range:

```bash
python data_puller.py --stock RELIANCE --from 2024-06-01 --to 2024-11-30
```

## Market Analyser

`market_analyser.py` calculates:

- IV Rank
- ATM implied volatility
- SMA20 and SMA50
- ADX
- Market trend
- Maximum call and put open-interest strikes
- Put-call ratio

Example:

```bash
python market_analyser.py HDFCBANK
```

## Strategy Scorer

`strategy_scorer.py` uses the market analysis to compare option strategies.

Current strategies include:

- Long straddle
- Long strangle
- Short strangle
- Iron condor

Example:

```bash
python strategy_scorer.py HDFCBANK
```

## Output

The tool can return:

- Selected strikes
- Option legs
- Premium paid or received
- Maximum profit
- Maximum loss
- Breakeven levels
- Estimated probability of profit

## Current Status

The project is still under development.

Before using it for real trades, the following still need more testing:

- Strategy calculations
- Strike selection
- Expiry handling
- Liquidity filters
- Brokerage and taxes
- Slippage
- Lot size and margin
- Backtesting

Options can involve large losses, especially for short positions. Every output should be checked manually.
