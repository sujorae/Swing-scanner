import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np


# ----------------------------------------------------
# Indicator helpers
# ----------------------------------------------------
def calculate_rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def clean_downloaded_data(df):
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


# ----------------------------------------------------
# Support, resistance, and watch zone helpers
# ----------------------------------------------------
def cluster_price_levels(levels, tolerance_pct=0.006):
    if not levels:
        return []

    levels = sorted(float(level) for level in levels if pd.notna(level))
    clusters = []

    for level in levels:
        if not clusters:
            clusters.append([level])
            continue

        cluster_average = sum(clusters[-1]) / len(clusters[-1])

        if abs(level - cluster_average) / cluster_average <= tolerance_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])

    clustered_levels = []

    for cluster in clusters:
        clustered_levels.append({
            "price": sum(cluster) / len(cluster),
            "touches": len(cluster)
        })

    return clustered_levels


def find_support_resistance(df, current_price, lookback=150, pivot_window=3, tolerance_pct=0.006):
    recent = df.tail(lookback).copy()

    if len(recent) < pivot_window * 2 + 1:
        return [], []

    lows = recent["Low"]
    highs = recent["High"]

    support_candidates = []
    resistance_candidates = []

    for i in range(pivot_window, len(recent) - pivot_window):
        low_window = lows.iloc[i - pivot_window:i + pivot_window + 1]
        high_window = highs.iloc[i - pivot_window:i + pivot_window + 1]

        current_low = float(lows.iloc[i])
        current_high = float(highs.iloc[i])

        if current_low <= float(low_window.min()):
            support_candidates.append(current_low)

        if current_high >= float(high_window.max()):
            resistance_candidates.append(current_high)

    clustered_supports = cluster_price_levels(support_candidates, tolerance_pct)
    clustered_resistances = cluster_price_levels(resistance_candidates, tolerance_pct)

    supports = [
        level for level in clustered_supports
        if level["price"] < current_price
    ]

    resistances = [
        level for level in clustered_resistances
        if level["price"] > current_price
    ]

    supports = sorted(
        supports,
        key=lambda level: abs(current_price - level["price"])
    )

    resistances = sorted(
        resistances,
        key=lambda level: abs(level["price"] - current_price)
    )

    return supports[:3], resistances[:3]


def format_zone(low_price, high_price):
    lower = min(low_price, high_price)
    upper = max(low_price, high_price)
    return f"{lower:.2f} to {upper:.2f}"


def add_percent_distance(level, current_price):
    if level is None or current_price == 0:
        return "n/a"

    distance = ((level - current_price) / current_price) * 100
    return f"{distance:+.2f}%"


def build_combined_levels(results, current_price):
    combined_supports = []
    combined_resistances = []

    for result in results:
        if result.get("error"):
            continue

        levels = result.get("levels", {})

        for support in levels.get("supports", []):
            combined_supports.append({
                "price": support["price"],
                "touches": support["touches"],
                "timeframe": result["label"]
            })

        for resistance in levels.get("resistances", []):
            combined_resistances.append({
                "price": resistance["price"],
                "touches": resistance["touches"],
                "timeframe": result["label"]
            })

    combined_supports = sorted(
        combined_supports,
        key=lambda level: abs(current_price - level["price"])
    )

    combined_resistances = sorted(
        combined_resistances,
        key=lambda level: abs(level["price"] - current_price)
    )

    return combined_supports, combined_resistances


def build_entry_watchlist(valid_results, current_price):
    combined_supports, combined_resistances = build_combined_levels(valid_results, current_price)

    daily_result = next((r for r in valid_results if r["label"] == "DAILY"), None)
    four_hour_result = next((r for r in valid_results if r["label"] == "4 HOUR"), None)
    five_min_result = next((r for r in valid_results if r["label"] == "5 MIN"), None)

    watchlist = []

    closest_support = combined_supports[0] if combined_supports else None
    closest_resistance = combined_resistances[0] if combined_resistances else None

    if closest_support:
        support_price = closest_support["price"]
        zone_low = support_price * 0.995
        zone_high = support_price * 1.005
        watchlist.append({
            "type": "Pullback watch",
            "price": format_zone(zone_low, zone_high),
            "note": f"Closest support from {closest_support['timeframe']} chart. This is a possible dip area if price pulls back and holds."
        })

    if closest_resistance:
        resistance_price = closest_resistance["price"]
        breakout_trigger = resistance_price * 1.003
        watchlist.append({
            "type": "Breakout watch",
            "price": f"above {breakout_trigger:.2f}",
            "note": f"Closest resistance from {closest_resistance['timeframe']} chart. A clean break above this area can show continuation."
        })

    if four_hour_result:
        ema9 = four_hour_result["ema9"]
        ema21 = four_hour_result["ema21"]
        ema_zone_low = min(ema9, ema21)
        ema_zone_high = max(ema9, ema21)

        if current_price >= ema_zone_high:
            watchlist.append({
                "type": "4 hour EMA retest watch",
                "price": format_zone(ema_zone_low, ema_zone_high),
                "note": "Price is above the 4 hour 9 EMA and 21 EMA zone. A controlled retest can be cleaner than chasing."
            })
        else:
            watchlist.append({
                "type": "4 hour EMA reclaim watch",
                "price": f"above {ema_zone_high:.2f}",
                "note": "Price is below the 4 hour EMA zone. Reclaiming this area would improve confirmation."
            })

    if daily_result:
        range_low = daily_result.get("range_20_low")
        range_high = daily_result.get("range_20_high")
        range_pct = daily_result.get("range_20_pct")

        if range_low and range_high and range_pct is not None and range_pct <= 12:
            watchlist.append({
                "type": "Possible accumulation zone",
                "price": format_zone(range_low, range_high),
                "note": "Recent daily range is fairly compressed. This can act like a consolidation area, but it still needs confirmation."
            })
        elif daily_result["ema50"]:
            watchlist.append({
                "type": "Daily trend support watch",
                "price": f"near {daily_result['ema50']:.2f}",
                "note": "The daily 50 EMA can act as a larger trend reference if price pulls back toward it."
            })

    if five_min_result:
        five_min_supports = five_min_result.get("levels", {}).get("supports", [])
        five_min_resistances = five_min_result.get("levels", {}).get("resistances", [])

        if five_min_supports and five_min_resistances:
            intraday_support = five_min_supports[0]["price"]
            intraday_resistance = five_min_resistances[0]["price"]
            watchlist.append({
                "type": "Intraday decision zone",
                "price": format_zone(intraday_support, intraday_resistance),
                "note": "Nearest 5 minute support and resistance. This is useful for timing, not for the whole trade thesis."
            })

    return watchlist, combined_supports, combined_resistances


# ----------------------------------------------------
# Timeframe scanner
# ----------------------------------------------------
def analyze_timeframe(ticker, label, period, interval):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    df = clean_downloaded_data(df)

    if df.empty or len(df) < 50:
        return {
            "label": label,
            "error": f"Not enough data returned for {label}."
        }

    open_price = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    rsi = calculate_rsi(close)
    macd_line, macd_signal, macd_hist = calculate_macd(close)

    current_price = float(close.iloc[-1])
    current_open = float(open_price.iloc[-1])
    current_ema9 = float(ema9.iloc[-1])
    current_ema21 = float(ema21.iloc[-1])
    current_ema50 = float(ema50.iloc[-1])
    current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
    previous_rsi = float(rsi.iloc[-2]) if pd.notna(rsi.iloc[-2]) else None
    current_macd_hist = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else None
    previous_macd_hist = float(macd_hist.iloc[-2]) if pd.notna(macd_hist.iloc[-2]) else None

    supports, resistances = find_support_resistance(
        df=df,
        current_price=current_price,
        lookback=150,
        pivot_window=3,
        tolerance_pct=0.006
    )

    range_20_low = float(low.tail(20).min())
    range_20_high = float(high.tail(20).max())
    range_20_pct = ((range_20_high - range_20_low) / current_price) * 100 if current_price else None

    # RSI logic:
    # Good confirmation is RSI above 50 but not extremely overbought.
    # RSI 45 to 50 gets partial credit if it is rising.
    rsi_bullish = current_rsi is not None and 50 <= current_rsi <= 70
    rsi_recovering = (
        current_rsi is not None
        and previous_rsi is not None
        and 45 <= current_rsi < 50
        and current_rsi > previous_rsi
    )

    # MACD logic:
    # Best case is positive histogram.
    # Partial case is negative histogram improving toward zero.
    macd_bullish = current_macd_hist is not None and current_macd_hist > 0
    macd_improving = (
        current_macd_hist is not None
        and previous_macd_hist is not None
        and current_macd_hist < 0
        and current_macd_hist > previous_macd_hist
    )

    # EMA logic:
    # This checks whether price is above short trend and whether short trend is above medium trend.
    ema_bullish = current_price > current_ema9 and current_ema9 > current_ema21
    ema_strong_trend = current_price > current_ema9 and current_ema9 > current_ema21 and current_ema21 > current_ema50

    # Candle and structure logic:
    # These are simple confirmation checks.
    green_candle = current_price > current_open
    higher_low = float(low.iloc[-1]) >= float(low.iloc[-2])
    higher_high = float(high.iloc[-1]) >= float(high.iloc[-2])
    above_ema50 = current_price > current_ema50

    checks = {
        "RSI": rsi_bullish or rsi_recovering,
        "MACD": macd_bullish or macd_improving,
        "EMA": ema_bullish,
        "Trend": ema_strong_trend or above_ema50,
        "Candle": green_candle,
        "Structure": higher_low or higher_high
    }

    # Scoring:
    # RSI, MACD, and EMA matter most.
    score = 0
    score += 2 if checks["RSI"] else 0
    score += 2 if checks["MACD"] else 0
    score += 2 if checks["EMA"] else 0
    score += 1 if checks["Trend"] else 0
    score += 1 if checks["Candle"] else 0
    score += 1 if checks["Structure"] else 0

    max_score = 9

    if score >= 7:
        rating = "YES"
    elif score >= 5:
        rating = "MAYBE"
    else:
        rating = "NO"

    return {
        "label": label,
        "error": None,
        "date": df.index[-1],
        "price": current_price,
        "rsi": current_rsi,
        "ema9": current_ema9,
        "ema21": current_ema21,
        "ema50": current_ema50,
        "macd_hist": current_macd_hist,
        "score": score,
        "max_score": max_score,
        "rating": rating,
        "checks": checks,
        "levels": {
            "supports": supports,
            "resistances": resistances
        },
        "range_20_low": range_20_low,
        "range_20_high": range_20_high,
        "range_20_pct": range_20_pct,
        "details": {
            "rsi_bullish": rsi_bullish,
            "rsi_recovering": rsi_recovering,
            "macd_bullish": macd_bullish,
            "macd_improving": macd_improving,
            "ema_bullish": ema_bullish,
            "ema_strong_trend": ema_strong_trend,
            "green_candle": green_candle,
            "higher_low": higher_low,
            "higher_high": higher_high,
            "above_ema50": above_ema50
        }
    }


def yes_no(value):
    return "YES" if value else "NO"


def format_rsi(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_float(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def run_scan(ticker):
    ticker = ticker.strip().upper()

    if ticker == "":
        ticker = "ORCL"

    timeframes = [
        {
            "label": "5 MIN",
            "period": "5d",
            "interval": "5m"
        },
        {
            "label": "4 HOUR",
            "period": "6mo",
            "interval": "1h"
        },
        {
            "label": "DAILY",
            "period": "1y",
            "interval": "1d"
        }
    ]

    results = []

    for timeframe in timeframes:
        result = analyze_timeframe(
            ticker=ticker,
            label=timeframe["label"],
            period=timeframe["period"],
            interval=timeframe["interval"]
        )
        results.append(result)

    valid_results = [r for r in results if not r.get("error")]

    if not valid_results:
        return f"No usable data returned for {ticker}. Check the ticker and try again."

    # Weighted final scoring:
    # 5 minute is entry timing.
    # 4 hour is the main swing setup.
    # Daily is the larger trend.
    weights = {
        "5 MIN": 1,
        "4 HOUR": 2,
        "DAILY": 2
    }

    weighted_score = 0
    weighted_max = 0

    for result in valid_results:
        weight = weights.get(result["label"], 1)
        weighted_score += result["score"] * weight
        weighted_max += result["max_score"] * weight

    final_percent = weighted_score / weighted_max if weighted_max else 0

    timeframe_ratings = {r["label"]: r["rating"] for r in valid_results}

    daily_ok = timeframe_ratings.get("DAILY") in ["YES", "MAYBE"]
    four_hour_ok = timeframe_ratings.get("4 HOUR") in ["YES", "MAYBE"]

    if final_percent >= 0.72 and daily_ok and four_hour_ok:
        final_status = "GOOD ENTRY SETUP"
        final_notes = "Daily and 4 hour agree enough to support an entry bias. The 5 minute can be used for entry timing."
    elif final_percent >= 0.58 and (daily_ok or four_hour_ok):
        final_status = "WATCHLIST ONLY"
        final_notes = "Some entry signals are there, but confirmation is not strong across enough timeframes yet."
    else:
        final_status = "NO ENTRY SETUP RIGHT NOW"
        final_notes = "The signals are too mixed or too weak for a clean entry right now."

    current_price = valid_results[-1]["price"]
    watchlist, combined_supports, combined_resistances = build_entry_watchlist(valid_results, current_price)

    output = []

    output.append("=" * 72)
    output.append(f"Ticker: {ticker}")
    output.append(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Current reference price: {current_price:.2f}")
    output.append("=" * 72)
    output.append("")
    output.append(f"FINAL STATUS: {final_status}")
    output.append(f"Score: {weighted_score}/{weighted_max} ({final_percent * 100:.1f}%)")
    output.append(final_notes)
    output.append("")

    output.append("KEY PRICE LEVELS")
    output.append("=" * 72)

    if combined_supports:
        output.append("Closest support levels below current price:")
        for level in combined_supports[:3]:
            output.append(
                f"  {level['price']:.2f}  "
                f"({level['timeframe']}, {add_percent_distance(level['price'], current_price)}, "
                f"{level['touches']} touch{'es' if level['touches'] != 1 else ''})"
            )
    else:
        output.append("Closest support levels below current price: none found")

    output.append("")

    if combined_resistances:
        output.append("Closest resistance levels above current price:")
        for level in combined_resistances[:3]:
            output.append(
                f"  {level['price']:.2f}  "
                f"({level['timeframe']}, {add_percent_distance(level['price'], current_price)}, "
                f"{level['touches']} touch{'es' if level['touches'] != 1 else ''})"
            )
    else:
        output.append("Closest resistance levels above current price: none found")

    output.append("")
    output.append("ENTRY WATCH AREAS")
    output.append("=" * 72)

    if watchlist:
        for item in watchlist:
            output.append(f"{item['type']}: {item['price']}")
            output.append(f"  {item['note']}")
            output.append("")
    else:
        output.append("No clean watch areas found from the current data.")
        output.append("")

    output.append("TIMEFRAME CONFIRMATION")
    output.append("=" * 72)
    output.append("Timeframe | Rating | RSI | MACD | EMA | Trend | Candle | Structure")
    output.append("-" * 72)

    for result in results:
        if result.get("error"):
            output.append(f"{result['label']:<9} | ERROR  | {result['error']}")
            continue

        checks = result["checks"]

        output.append(
            f"{result['label']:<9} | "
            f"{result['rating']:<6} | "
            f"{yes_no(checks['RSI']):<3} | "
            f"{yes_no(checks['MACD']):<5} | "
            f"{yes_no(checks['EMA']):<3} | "
            f"{yes_no(checks['Trend']):<5} | "
            f"{yes_no(checks['Candle']):<6} | "
            f"{yes_no(checks['Structure'])}"
        )

    output.append("")
    output.append("DETAILS BY TIMEFRAME")
    output.append("=" * 72)

    for result in results:
        output.append("")
        output.append(result["label"])
        output.append("-" * 72)

        if result.get("error"):
            output.append(result["error"])
            continue

        details = result["details"]

        output.append(f"Price: {result['price']:.2f}")
        output.append(f"RSI: {format_rsi(result['rsi'])}")
        output.append(f"9 EMA: {format_float(result['ema9'])}")
        output.append(f"21 EMA: {format_float(result['ema21'])}")
        output.append(f"50 EMA: {format_float(result['ema50'])}")
        output.append(f"MACD histogram: {result['macd_hist']:.4f}")
        output.append(f"20 bar range: {result['range_20_low']:.2f} to {result['range_20_high']:.2f} ({result['range_20_pct']:.2f}%)")
        output.append(f"Score: {result['score']}/{result['max_score']}")
        output.append("")

        supports = result["levels"]["supports"]
        resistances = result["levels"]["resistances"]

        if supports:
            support_text = ", ".join([f"{level['price']:.2f}" for level in supports])
        else:
            support_text = "none"

        if resistances:
            resistance_text = ", ".join([f"{level['price']:.2f}" for level in resistances])
        else:
            resistance_text = "none"

        output.append(f"Nearest supports: {support_text}")
        output.append(f"Nearest resistances: {resistance_text}")
        output.append("")
        output.append(f"RSI bullish or recovering: {yes_no(result['checks']['RSI'])}")
        output.append(f"MACD bullish or improving: {yes_no(result['checks']['MACD'])}")
        output.append(f"Price and EMA trend bullish: {yes_no(result['checks']['EMA'])}")
        output.append(f"Above larger trend support: {yes_no(result['checks']['Trend'])}")
        output.append(f"Current candle green: {yes_no(result['checks']['Candle'])}")
        output.append(f"Higher low or higher high: {yes_no(result['checks']['Structure'])}")
        output.append("")
        output.append("Sub checks:")
        output.append(f"RSI 50 to 70: {yes_no(details['rsi_bullish'])}")
        output.append(f"RSI 45 to 50 and rising: {yes_no(details['rsi_recovering'])}")
        output.append(f"MACD histogram positive: {yes_no(details['macd_bullish'])}")
        output.append(f"MACD histogram negative but improving: {yes_no(details['macd_improving'])}")
        output.append(f"Price > 9 EMA and 9 EMA > 21 EMA: {yes_no(details['ema_bullish'])}")
        output.append(f"9 EMA > 21 EMA > 50 EMA: {yes_no(details['ema_strong_trend'])}")
        output.append(f"Price above 50 EMA: {yes_no(details['above_ema50'])}")

    output.append("")
    output.append("HOW TO READ THIS")
    output.append("=" * 72)
    output.append("Daily = larger trend direction.")
    output.append("4 hour = main swing setup confirmation.")
    output.append("5 minute = entry timing, not the main reason to take the trade.")
    output.append("")
    output.append("A cleaner entry usually has DAILY and 4 HOUR showing YES or MAYBE.")
    output.append("Support and resistance are estimated from recent pivot highs and pivot lows.")
    output.append("Watch areas are zones to monitor, not automatic buy signals.")
    output.append("")
    output.append("This scanner does not guarantee a winning trade. It only checks whether the setup is technically aligned.")

    return "\n".join(output)


# ----------------------------------------------------
# GUI
# ----------------------------------------------------
def scan_button_clicked():
    ticker = ticker_entry.get()

    results_box.config(state="normal")
    results_box.delete("1.0", tk.END)
    results_box.insert(tk.END, "Scanning multiple timeframes. Give it a few seconds...")
    results_box.config(state="disabled")

    scan_button.config(state="disabled")

    def task():
        try:
            result = run_scan(ticker)
        except Exception as e:
            result = f"Error:\n{e}"

        results_box.config(state="normal")
        results_box.delete("1.0", tk.END)
        results_box.insert(tk.END, result)
        results_box.config(state="disabled")
        scan_button.config(state="normal")

    threading.Thread(target=task, daemon=True).start()


root = tk.Tk()
root.title("Swing Scanner")
root.geometry("980x760")
root.configure(bg="#111827")

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TLabel",
    background="#111827",
    foreground="#E5E7EB",
    font=("Segoe UI", 11)
)

style.configure(
    "Header.TLabel",
    background="#111827",
    foreground="#F9FAFB",
    font=("Segoe UI", 22, "bold")
)

style.configure(
    "TButton",
    font=("Segoe UI", 11, "bold"),
    padding=8
)

style.configure(
    "TEntry",
    font=("Segoe UI", 12),
    padding=8
)

main_frame = tk.Frame(root, bg="#111827")
main_frame.pack(fill="both", expand=True, padx=24, pady=24)

title_label = ttk.Label(
    main_frame,
    text="Multi Timeframe Swing Scanner",
    style="Header.TLabel"
)
title_label.pack(anchor="w")

subtitle_label = ttk.Label(
    main_frame,
    text="Checks entry confirmation, price levels, and watch zones across multiple timeframes."
)
subtitle_label.pack(anchor="w", pady=(4, 20))

input_frame = tk.Frame(main_frame, bg="#111827")
input_frame.pack(fill="x", pady=(0, 16))

ticker_label = ttk.Label(input_frame, text="Ticker:")
ticker_label.pack(side="left")

ticker_entry = ttk.Entry(input_frame, width=15)
ticker_entry.pack(side="left", padx=(8, 12))
ticker_entry.insert(0, "ORCL")

scan_button = ttk.Button(
    input_frame,
    text="Scan",
    command=scan_button_clicked
)
scan_button.pack(side="left")

results_frame = tk.Frame(main_frame, bg="#111827")
results_frame.pack(fill="both", expand=True)

results_box = tk.Text(
    results_frame,
    wrap="word",
    bg="#020617",
    fg="#E5E7EB",
    insertbackground="#E5E7EB",
    font=("Consolas", 11),
    padx=16,
    pady=16,
    relief="flat"
)
results_box.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(results_frame, command=results_box.yview)
scrollbar.pack(side="right", fill="y")
results_box.config(yscrollcommand=scrollbar.set)

results_box.insert(tk.END, "Enter a ticker and click Scan.")
results_box.config(state="disabled")

root.mainloop()
