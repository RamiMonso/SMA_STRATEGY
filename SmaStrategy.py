# app.py
# SMA Backtester - Streamlit App
# הפעלה דרך GitHub + Streamlit Cloud
# דרישות: yfinance, pandas, numpy, plotly, matplotlib

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="SMA Backtester", layout="wide")

# ---------------------------------------------------------------------
# פונקציות עזר
# ---------------------------------------------------------------------

def fetch_data(ticker, start_fetch, end_fetch, interval='1d'):
    tk = yf.Ticker(ticker)
    df = tk.history(start=start_fetch, end=end_fetch + timedelta(days=1), interval=interval)
    if df.empty:
        return df
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    else:
        df['Adj_Close'] = df['Close']
    return df


def compute_sma(df, period):
    return df['Adj_Close'].rolling(window=period, min_periods=period).mean()


def run_backtest(df, start_date, end_date, sma_period, pct_threshold,
                 fee_entry, fee_exit, fee_type, invest_mode,
                 fixed_invest, start_capital, close_open_with_today):

    df_local = df.copy()
    df_local['SMA'] = compute_sma(df_local, sma_period)
    df_period = df_local.loc[(df_local.index.date >= start_date) &
                             (df_local.index.date <= end_date)]

    trades = []
    position = None
    capital = float(start_capital)

    for date, row in df_period.iterrows():
        price = float(row['Adj_Close'])
        sma = row['SMA']
        if np.isnan(sma):
            continue

        entry_threshold = sma * (1 + pct_threshold / 100.0)
        exit_threshold = sma * (1 - pct_threshold / 100.0)

        # כניסה
        if position is None and price >= entry_threshold:
            invest_amount = fixed_invest if invest_mode == 'fixed_per_trade' else capital
            if invest_amount <= 0:
                break
            fee_e = invest_amount * fee_entry / 100 if fee_type == 'percent' else fee_entry
            shares = max(invest_amount - fee_e, 0) / price
            position = {
                'entry_date': date.date(),
                'entry_price': price,
                'entry_sma': sma,
                'entry_fee': fee_e,
                'shares': shares,
                'invested': invest_amount,
            }

        # יציאה
        elif position and price <= exit_threshold:
            fee_x = (position['shares'] * price) * fee_exit / 100 if fee_type == 'percent' else fee_exit
            proceeds = position['shares'] * price - fee_x
            pnl = proceeds - position['invested']
            pct = (proceeds / position['invested'] - 1) * 100
            if invest_mode == 'compound':
                capital = proceeds
            trades.append({
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'entry_sma': position['entry_sma'],
                'exit_date': date.date(),
                'exit_price': price,
                'exit_sma': sma,
                'pnl': pnl,
                'pnl_pct': pct
            })
            position = None

    # פוזיציה פתוחה
    if position and close_open_with_today:
        last_price = df_local.iloc[-1]['Adj_Close']
        last_sma = df_local.iloc[-1]['SMA']
        fee_x = (position['shares'] * last_price) * fee_exit / 100 if fee_type == 'percent' else fee_exit
        proceeds = position['shares'] * last_price - fee_x
        pnl = proceeds - position['invested']
        pct = (proceeds / position['invested'] - 1) * 100
        trades.append({
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'entry_sma': position['entry_sma'],
            'exit_date': df_local.index[-1].date(),
            'exit_price': last_price,
            'exit_sma': last_sma,
            'pnl': pnl,
            'pnl_pct': pct
        })

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------
# ממשק משתמש
# ---------------------------------------------------------------------

st.title("📊 SMA Backtester — בדיקת אסטרטגיית ממוצעים נעים (SMA)")

with st.sidebar:
    st.header("הגדרות")
    ticker = st.text_input("שם מניה (Ticker):", "AAPL")
    start_date = st.date_input("תאריך התחלה", datetime.today() - timedelta(days=365))
    end_date = st.date_input("תאריך סיום", datetime.today())
    sma_period = st.number_input("מספר ימים לחישוב SMA", 2, 1000, 50)
    pct_threshold = st.number_input("הפרש באחוזים בין המחיר לממוצע", -100.0, 1000.0, 0.0)
    fee_type = st.selectbox("סוג עמלה", ["percent", "fixed"])
    fee_entry = st.number_input("עמלת כניסה", 0.0)
    fee_exit = st.number_input("עמלת יציאה", 0.0)
    invest_mode = st.selectbox("שיטת השקעה", ["fixed_per_trade", "compound"])
    fixed_invest = st.number_input("סכום קבוע לכל עסקה", 1000.0)
    start_capital = st.number_input("הון התחלתי", 1000.0)
    close_open_with_today = st.checkbox("לסגור פוזיציה פתוחה ביום האחרון", True)
    interval = st.selectbox("תדירות נתונים", ["1d", "1h"], index=0)
    run_button = st.button("הרץ בדיקה")

if run_button:
    st.info("מוריד נתונים מ-Yahoo Finance...")
    fetch_start = start_date - timedelta(days=250)
    fetch_end = datetime.today().date() if close_open_with_today else end_date
    df = fetch_data(ticker, fetch_start, fetch_end, interval)
    if df.empty:
        st.error("לא נמצאו נתונים. בדוק את הסימול.")
    else:
        trades = run_backtest(df, start_date, end_date, sma_period, pct_threshold,
                              fee_entry, fee_exit, fee_type, invest_mode,
                              fixed_invest, start_capital, close_open_with_today)
        if trades.empty:
            st.warning("לא נמצאו פוזיציות בתקופה זו.")
        else:
            st.subheader("📅 טבלת עסקאות")
            st.dataframe(trades)

            # סיכום
            total_pnl = trades["pnl"].sum()
            total_pct = trades["pnl_pct"].mean()
            st.metric("רווח/הפסד כולל", f"{round(total_pnl, 2)}")
            st.metric("תשואה ממוצעת (%)", f"{round(total_pct, 2)}")

            # גרף
            st.subheader("📈 גרף מחיר + SMA + עסקאות")
            df["SMA"] = compute_sma(df, sma_period)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Adj_Close"], name="מחיר מתואם"))
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], name=f"SMA {sma_period}"))
            for _, t in trades.iterrows():
                fig.add_trace(go.Scatter(x=[t["entry_date"]], y=[t["entry_price"]],
                                         mode="markers", marker_symbol="triangle-up", name="כניסה"))
                fig.add_trace(go.Scatter(x=[t["exit_date"]], y=[t["exit_price"]],
                                         mode="markers", marker_symbol="triangle-down", name="יציאה"))
            fig.update_layout(height=600, xaxis_title="תאריך", yaxis_title="מחיר")
            st.plotly_chart(fig, use_container_width=True)

            # הורדת טבלה
            csv = trades.to_csv(index=False).encode("utf-8")
            st.download_button("📥 הורד טבלה (CSV)", csv, "trades.csv", "text/csv")
