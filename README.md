# Swing Scanner

Swing Scanner is a Python desktop app that checks whether a stock has a clean setup across multiple timeframes. It looks at the 5 minute, 4 hour, and daily charts to compare short term entry timing with the larger trend.

The app uses RSI, MACD, EMA trend structure, candle direction, and basic price structure to give each timeframe a rating of YES, MAYBE, or NO. It then gives a final status for whether the ticker looks like a good long setup, a watchlist setup, or no setup right now.

## Features

Multi timeframe analysis using 5 minute, 4 hour, and daily data

Ticker input field for checking any stock symbol

RSI confirmation and recovery checks

MACD histogram confirmation and improvement checks

EMA trend checks using 9 EMA, 21 EMA, and 50 EMA

Simple desktop interface built with Tkinter

Standalone executable built with PyInstaller

## Tech Stack

Python

Tkinter

yfinance

pandas

NumPy

PyInstaller

## How It Works

The scanner downloads recent market data for the selected ticker using yfinance. It calculates RSI, MACD, and EMA values for each timeframe. Each timeframe receives a score based on whether the technical indicators support a long entry.

The daily chart is used for larger trend direction.

The 4 hour chart is used for the main swing setup.

The 5 minute chart is used for short term entry timing.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python swing_app.py
```

## Disclaimer

This project is for educational and personal analysis purposes only. It does not provide financial advice or guarantee profitable trades.
