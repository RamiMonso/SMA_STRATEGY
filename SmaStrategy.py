# app.py
# SMA Backtester with threshold crossover entry
# For Streamlit Cloud (GitHub -> Streamlit Cloud)
# Requirements: streamlit, yfinance, pandas, numpy, plotly, matplotlib

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="SMA Backtester (Crossover Entry)", layout="wide")

# ---------------------- Helpers ----------------------

def fetch_data(ticker, start_fetch, end_fetch, interval='1d'):
    tk = yf.Ticker(ticker)
    # yfinance expects strings or dates; ensure end is inclusive by adding one day when daily
    df = tk.history(start=start_fetch, end=(end_fetch + timedelta(days=1)) if isinstance(end_fetch, datetime) or isinstance(end_fetch, pd.Timestamp) else end_fetch, interval=interval)
    if df.empty:
        return df
    # Normalize column name
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Adj_Close'})
    else:
        df['Adj_Close'] = df['Close']
    df.index = pd.to_datetime(df.index)
    return df

def compute_sma(df, period):
    return df['Adj_Close'].rolling(window=period, min_periods=period).mean()

def run_backtest(df, start_date, end_date, sma_period, pct_threshold,
                 fee_entry, fee_exit, fee_type, invest_mode,
                 fixed_invest, start_capital, close_open_with_today):
    """
    Entry rule: requires previous trading day price < prev_SMA*(1+threshold%)
                AND current price >= SMA*(1+threshold%).
    Exit rule: price <= SMA*(1-threshold%).
    """
    df_local = df.copy().sort_index()
    df_local['SMA'] = compute_sma(df_local, sma_period)

    # restrict to full available range but we will iterate using indices in df_local
    # determine mask for evaluation period
    mask = (df_local.index.date >= start_date) & (df_local.index.date <= end_date)
    df_period = df_local.loc[mask]

    trades = []
    position = None
    capital = float(start_capital)

    # Make sure we can map each period index to position in df_local
    all_index = df_local.index

    for curr_idx in df_period.index:
        # find integer position of current index in full df
        pos = all_index.get_loc(curr_idx)
        if pos == 0:
            # no previous trading day available => skip (can't detect crossover)
            continue
        prev_pos = pos - 1
        prev_idx = all_index[prev_pos]

        price = float(df_local.loc[curr_idx, 'Adj_Close'])
        sma = df_local.loc[curr_idx, 'SMA']
        prev_price = float(df_local.loc[prev_idx, 'Adj_Close'])
        prev_sma = df_local.loc[prev_idx, 'SMA']

        # skip if SMA not defined (warmup)
        if np.isnan(sma) or np.isnan(prev_sma):
            continue

        # thresholds
        up_threshold_curr = sma * (1 + pct_threshold / 100.0)
        up_threshold_prev = prev_sma * (1 + pct_threshold / 100.0)
        down_threshold_curr = sma * (1 - pct_threshold / 100.0)

        # ENTRY: only if previous day was below prev up-threshold AND current day is >= current up-threshold
        if position is None:
            crossed_up = (prev_price < up_threshold_prev) and (price >= up_threshold_curr)
            if crossed_up:
                # determine invest amount
                invest_amount = fixed_invest if invest_mode == 'fixed_per_trade' else capital
                if invest_amount <= 0:
                    # nothing to invest
                    continue
                fee_e = invest_amount * fee_entry / 100.0 if fee_type == 'percent' else fee_entry
                cash_for_shares = max(invest_amount - fee_e, 0.0)
                shares = cash_for_shares / price if price > 0 else 0.0
                position = {
                    'entry_date': curr_idx.date(),
                    'entry_price': price,
                    'entry_sma': sma,
                    'entry_fee': fee_e,
                    'shares': shares,
                    'invested': invest_amount
                }

        # EXIT: if there is an open position and price <= down_threshold_curr
        if position is not None:
            if price <= down_threshold_curr:
                fee_x = (position['shares'] * price) * fee_exit / 100.0 if fee_type == 'percent' else fee_exit
                proceeds = position['shares'] * price - fee_x
                pnl = proceeds - position['invested']
                pnl_pct = (proceeds / position['invested'] - 1) * 100 if position['invested'] != 0 else 0.0
                if invest_mode == 'compound':
                    capital = proceeds
                trades.append({
                    'entry_date': position['entry_date'],
                    'entry_price': position['entry_price'],
                    'entry_sma': position['entry_sma'],
                    'exit_date': curr_idx.date(),
                    'exit_price': price,
                    'exit_sma': sma,
                    'entry_fee': position['entry_fee'],
                    'exit_fee': fee_x,
                    'invested': position['invested'],
                    'shares': position['shares'],
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                position = None

    # If still open at the end of data / user requested to close by last available price
    if position is not None:
        if close_open_with_today:
            last_price = float(df_local.iloc[-1]['Adj_Close'])
            last_sma = float(df_local.iloc[-1]['SMA']) if not np.isnan(df_local.iloc[-1]['SMA']) else None
            fee_x = (position['shares'] * last_price) * fee_exit / 100.0 if fee_type == 'percent' else fee_exit
            proceeds = position['shares'] * last_price - fee_x
            pnl = proceeds - position['invested']
            pnl_pct = (proceeds / position['invested'] - 1) * 100 if position['invested'] != 0 else 0.0
            trades.append({
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'entry_sma': position['entry_sma'],
                'exit_date': df_local.index[-1].date(),
                'exit_price': last_price,
                'exit_sma': last_sma,
                'entry_fee': position['entry_fee'],
                'exit_fee': fee_x,
                'invested': position['invested'],
                'shares': position['shares'],
                'proceeds': proceeds,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'closed_by': 'closed_at_last'
            })
        else:
            # report as open (no exit fields)
            trades.append({
                'entry_date': position['entry_date'],
                'entry_price': position['entry_price'],
                'entry_sma': position['entry_sma'],
                'exit_date': None,
                'exit_price': None,
                'exit_sma': None,
                'entry_fee': position['entry_fee'],
                'exit_fee': None,
                'invested': position['invested'],
                'shares': position['shares'],
                'proceeds': None,
                'pnl': None,
                'pnl_pct': None,
                'closed_by': 'open'
            })

    trades_df = pd.DataFrame(trades)
    return trades_df

# ---------------------- UI ----------------------

st.title("📊 SMA Backtester — Crossover Entry (threshold)")

with st.sidebar:
    st.header("הגדרות בדיקה")
    ticker = st.text_input("שם מניה (Ticker) — לדוגמה AAPL", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("תאריך התחלה", datetime.today().date() - timedelta(days=365))
    with col2:
        end_date = st.date_input("תאריך סיום", datetime.today().date())
    sma_period = st.number_input("מספר ימים לחישוב SMA", min_value=2, max_value=2000, value=50)
    pct_threshold = st.number_input("הפרש באחוזים (threshold) — כניסה/יציאה", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

    st.markdown("---")
    st.header("עמלות")
    fee_type = st.selectbox("סוג עמלה", options=["percent", "fixed"], format_func=lambda x: "אחוזים" if x=="percent" else "סכום קבוע")
    fee_entry = st.number_input("עמלת כניסה", value=0.0)
    fee_exit = st.number_input("עמלת יציאה", value=0.0)

    st.markdown("---")
    st.header("אופציות השקעה")
    invest_mode = st.selectbox("שיטת החישוב", options=["fixed_per_trade", "compound"], format_func=lambda x: "פיקדון קבוע לכל עסקה" if x=="fixed_per_trade" else "ריבית דריבית / מצטברת")
    fixed_invest = st.number_input("סכום קבוע לכל עסקה (אם נבחר)", value=1000.0)
    start_capital = st.number_input("הון התחלתי (למצב מצטבר)", value=1000.0)

    st.markdown("---")
    st.header("הגדרות נוספות")
    close_open_with_today = st.checkbox("אם פוזיציה פתוחה — לסגור לפי המחיר האחרון הזמין", value=True)
    include_fees_in_buy_hold = st.checkbox("להחיל עמלות גם על BUY & HOLD", value=False)
    interval = st.selectbox("תדירות נתונים", options=["1d", "1h"], index=0, format_func=lambda x: "יומי" if x=="1d" else "שעתי")

    run_button = st.button("הרץ בדיקה")

if run_button:
    st.info("מוריד נתונים מ-Yahoo Finance...")
    # Warmup 250 days (get extra days before start to compute SMA)
    warmup_days = 250
    fetch_start = start_date - timedelta(days=warmup_days * 2)
    fetch_end = datetime.today().date() if close_open_with_today else end_date
    df = fetch_data(ticker, fetch_start, fetch_end, interval=interval)

    if df.empty:
        st.error("לא נמצאו נתונים — בדוק את סימול המניה ותדירות הנתונים (שעתי עשוי להיות חסר עבור תקופות ארוכות).")
    else:
        # Ensure Adj_Close exists
        if 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        df['SMA'] = compute_sma(df, sma_period)

        trades_df = run_backtest(df, start_date, end_date, sma_period, pct_threshold,
                                 fee_entry, fee_exit, fee_type, invest_mode,
                                 fixed_invest, start_capital, close_open_with_today)

        st.subheader("טבלת פוזיציות")
        if trades_df.empty:
            st.info("לא זוהו פוזיציות בתקופה שנבחרה.")
        else:
            display_df = trades_df.copy()
            # Select and format columns for display
            cols = ["entry_date","entry_sma","entry_price","exit_date","exit_sma","exit_price","pnl_pct"]
            # Some columns may not exist (e.g., pnl_pct) — guard
            for c in ["entry_sma","entry_price","exit_sma","exit_price","pnl_pct"]:
                if c in display_df.columns:
                    display_df[c] = display_df[c].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
            # Rename to Hebrew
            rename_map = {
                "entry_date":"תאריך כניסה",
                "entry_sma":"SMA ביום כניסה",
                "entry_price":"מחיר כניסה",
                "exit_date":"תאריך יציאה",
                "exit_sma":"SMA ביום יציאה",
                "exit_price":"מחיר יציאה",
                "pnl_pct":"% רווח/הפסד"
            }
            # Only keep columns present
            present_cols = [c for c in rename_map.keys() if c in display_df.columns]
            st.dataframe(display_df[present_cols].rename(columns=rename_map), use_container_width=True)

            # CSV download
            csv = trades_df.to_csv(index=False).encode("utf-8")
            st.download_button("הורד טבלה (CSV)", data=csv, file_name=f"trades_{ticker}_{start_date}_{end_date}.csv", mime="text/csv")

        # Summary
        st.subheader("סיכום ביצועים")
        if not trades_df.empty and 'pnl' in trades_df.columns:
            total_pnl = trades_df['pnl'].dropna().sum()
            # overall pct depending on mode
            total_pct = None
            if invest_mode == 'compound' and not trades_df.empty:
                final_cap = trades_df.iloc[-1]['proceeds'] if pd.notnull(trades_df.iloc[-1]['proceeds']) else start_capital
                total_pct = (final_cap / start_capital - 1) * 100
            elif invest_mode == 'fixed_per_trade':
                invested_total = trades_df['invested'].dropna().sum()
                proceeds_total = trades_df['proceeds'].dropna().sum() if 'proceeds' in trades_df.columns else None
                if proceeds_total is not None and invested_total > 0:
                    total_pct = (proceeds_total / invested_total - 1) * 100

            colA, colB = st.columns(2)
            with colA:
                st.metric("סך רווח/הפסד (מטבע)", f"{round(total_pnl, 2)}")
            with colB:
                st.metric("סך רווח/הפסד (%)", f"{round(total_pct, 2) if total_pct is not None else 'N/A'}")
        else:
            st.write("אין עסקאות לצורך חישוב סיכום.")

        # Buy & Hold
        st.subheader("השוואה ל-BUY & HOLD")
        try:
            # find first available price on/after start_date and last on/before end_date (or last available)
            df_for_bh = df.loc[(df.index.date >= start_date) & (df.index.date <= (end_date if not close_open_with_today else df.index[-1].date()))]
            bh_start_price = float(df_for_bh.iloc[0]['Adj_Close'])
            bh_end_price = float(df_for_bh.iloc[-1]['Adj_Close'])
            bh_return = (bh_end_price / bh_start_price - 1) * 100
            if include_fees_in_buy_hold:
                if fee_type == 'percent':
                    fee_e = bh_start_price * fee_entry / 100.0
                    fee_x = bh_end_price * fee_exit / 100.0
                else:
                    fee_e = fee_entry
                    fee_x = fee_exit
                bh_return = ((bh_end_price - fee_x) / (bh_start_price + fee_e) - 1) * 100
            st.write(f"BUY & HOLD — התחלה: {round(bh_start_price,4)}   סוף: {round(bh_end_price,4)}   תשואה: {round(bh_return,4)} %")
        except Exception:
            st.write("לא ניתן לחשב BUY & HOLD (חוסר בנתונים לטווח המתאים).")

        # Plot chart with markers
        st.subheader("גרף מחיר מתואם + SMA + כניסות/יציאות")
        df_plot = df.copy()
        df_plot['SMA'] = compute_sma(df_plot, sma_period)
        # display region: from 30 days before start to end/last
        plot_start = (pd.to_datetime(start_date) - timedelta(days=30)).date()
        plot_end = df_plot.index[-1].date() if close_open_with_today else end_date
        mask_plot = (df_plot.index.date >= plot_start) & (df_plot.index.date <= plot_end)
        df_plot = df_plot.loc[mask_plot]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Adj_Close'], name='Adj Close'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA'], name=f'SMA {sma_period}'))

        if not trades_df.empty:
            for _, t in trades_df.iterrows():
                if pd.notnull(t.get('entry_date')):
                    ed = pd.to_datetime(t['entry_date'])
                    if ed in df_plot.index:
                        fig.add_trace(go.Scatter(x=[ed], y=[df_plot.loc[ed]['Adj_Close']], mode='markers', marker=dict(symbol='triangle-up', size=12), name='Entry'))
                if pd.notnull(t.get('exit_date')):
                    xd = pd.to_datetime(t['exit_date'])
                    if xd in df_plot.index:
                        fig.add_trace(go.Scatter(x=[xd], y=[df_plot.loc[xd]['Adj_Close']], mode='markers', marker=dict(symbol='triangle-down', size=12), name='Exit'))

        fig.update_layout(height=600, xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

        # provide PNG download (matplotlib)
        buf = io.BytesIO()
        plt.figure(figsize=(12,6))
        plt.plot(df_plot.index, df_plot['Adj_Close'])
        plt.plot(df_plot.index, df_plot['SMA'])
        if not trades_df.empty:
            for _, t in trades_df.iterrows():
                if pd.notnull(t.get('entry_date')):
                    ed = pd.to_datetime(t['entry_date'])
                    if ed in df_plot.index:
                        plt.scatter(ed, df_plot.loc[ed]['Adj_Close'], marker='^')
                if pd.notnull(t.get('exit_date')):
                    xd = pd.to_datetime(t['exit_date'])
                    if xd in df_plot.index:
                        plt.scatter(xd, df_plot.loc[xd]['Adj_Close'], marker='v')
        plt.title(f"{ticker} Adj Close + SMA {sma_period}")
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        st.download_button('הורד גרף PNG', data=buf, file_name=f"chart_{ticker}_{start_date}_{end_date}.png", mime='image/png')

        st.success("בדיקה הושלמה")

# Footer notes
st.markdown("---")
st.write("""
**הערות חשובות:**
- כניסה מבוצעת רק כאשר יש חציית מצב מ־below → above לפי הסף (כלומר: אתמול היה < prev_SMA*(1+threshold), והיום >= SMA*(1+threshold)).
- יציאה מופעלת כאשר price <= SMA*(1-threshold).
- הנתונים נמשכים מ-Yahoo Finance ומשתמשים ב-Adj Close כשזמין.
- חימום SMA: מורידים ימים נוספים לפני תאריך ההתחלה (ברירת מחדל: 250 ימים) כדי לאפשר חישוב תקין של ה-SMA.
- זמינות נתוני שעתי ותוקפם ב-yfinance מוגבליים לתקופות קצרות יותר מאשר נתוני יומי.
""")
