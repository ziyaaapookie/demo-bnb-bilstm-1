import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# ==============================
# KONFIGURASI
# ==============================
WINDOW_SIZE = 60          # harus sama dengan waktu training
EVAL_DAYS = 7             # 7 hari ke belakang untuk evaluasi
DEFAULT_FORECAST_DAYS = 7 # default horizon ke depan
TICKER = "BNB-USD"

ARTEFACT_DIR = "artefak_bnb"
LSTM_PATH = os.path.join(ARTEFACT_DIR, "lstm_bnb.h5")
BILSTM_PATH = os.path.join(ARTEFACT_DIR, "bilstm_bnb.h5")
SCALER_PATH = os.path.join(ARTEFACT_DIR, "scaler_bnb.pkl")

BNB_LOGO_URL = "https://i.ibb.co.com/8DGZs0r9/pngwing-com-1.png"


# ==============================
# FUNGSI BANTU
# ==============================
def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE custom (hindari pembagian dengan nol)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0


def create_sliding_window(data, window_size):
    """Buat X, y dengan skema sliding window dari data 1D (scaled)."""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def hari_indo_from_date(d):
    if isinstance(d, pd.Timestamp):
        d = d.date()
    hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    return hari_list[d.weekday()]


def format_tanggal_indo(d):
    if isinstance(d, pd.Timestamp):
        d = d.date()
    bulan_list = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                  "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
    hari = hari_indo_from_date(d)
    bulan = bulan_list[d.month - 1]
    return f"{hari}, {d.day:02d} {bulan} {d.year}"


@st.cache_data(show_spinner=False, ttl=600)
def load_daily_data(ticker=TICKER):
    """
    Ambil data harian BNB (1d) dari Yahoo Finance.
    Output: df dengan kolom ['Date', 'Price'] terurut naik.
    Robust kalau kolom 'Close' beda nama / multiindex.
    """
    try:
        daily = yf.download(
            ticker,
            start="2020-01-01",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data harian dari Yahoo Finance: {e}")

    if daily is None or len(daily) == 0:
        raise RuntimeError("Data harian kosong dari Yahoo Finance.")

    # Pastikan DataFrame
    if isinstance(daily, pd.Series):
        daily = daily.to_frame()

    # Flatten kalau kolom multiindex
    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = [
            "_".join([str(c) for c in col if str(c) != ""]).strip("_")
            for col in daily.columns
        ]

    cols = list(daily.columns)

    # Cari kolom harga yang paling masuk akal
    price_col = None
    priority = ["Close", "Adj Close", "close", "adjclose", "AdjClose", "Adj_Close"]
    for p in priority:
        if p in cols:
            price_col = p
            break

    # Kalau belum ketemu, ambil kolom numerik pertama yang bukan volume/dll
    if price_col is None:
        numeric_cols = [
            c for c in cols
            if np.issubdtype(daily[c].dtype, np.number)
            and c.lower() not in ["volume", "dividends", "stock splits", "stocksplits"]
        ]
        if not numeric_cols:
            raise RuntimeError(f"Tidak menemukan kolom harga numerik. Kolom: {cols}")
        price_col = numeric_cols[0]

    df = daily[[price_col]].copy().dropna(subset=[price_col])
    df = df.reset_index()

    # Cari kolom tanggal
    date_col = None
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            date_col = col
            break
    if date_col is None:
        raise RuntimeError(f"Tidak menemukan kolom datetime. Kolom: {list(df.columns)}")

    df = df.rename(columns={date_col: "Date", price_col: "Price"})
    df = df.dropna(subset=["Date", "Price"]).reset_index(drop=True)

    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = df["Price"].astype(float)

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_model_no_compile(path):
    """
    Load model Keras/TensorFlow tanpa compile
    untuk menghindari error deserialisasi metrics.
    """
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e1:
        try:
            from keras.saving import load_model as keras_load_model
            model = keras_load_model(path, compile=False, safe_mode=False)
            return model
        except Exception as e2:
            raise RuntimeError(
                f"Gagal memuat model dari {path}.\n"
                f"Error tf.keras: {e1}\n"
                f"Error keras.saving: {e2}"
            )


@st.cache_resource(show_spinner=False)
def load_models_and_scaler():
    """Load LSTM, Bi-LSTM dan scaler dari artefak_bnb."""
    if not os.path.exists(ARTEFACT_DIR):
        raise FileNotFoundError(
            f"Folder '{ARTEFACT_DIR}' tidak ditemukan. "
            f"Pastikan berada di direktori yang sama dengan app.py."
        )

    if not os.path.exists(LSTM_PATH):
        raise FileNotFoundError(f"Model LSTM tidak ditemukan di {LSTM_PATH}")
    if not os.path.exists(BILSTM_PATH):
        raise FileNotFoundError(f"Model Bi-LSTM tidak ditemukan di {BILSTM_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler tidak ditemukan di {SCALER_PATH}")

    lstm_model = load_model_no_compile(LSTM_PATH)
    bilstm_model = load_model_no_compile(BILSTM_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return lstm_model, bilstm_model, scaler


def prepare_data(df, scaler, window_size=WINDOW_SIZE):
    """
    - Ambil kolom Price
    - Scale pakai scaler (hasil training)
    - Buat sliding window
    """
    prices = df["Price"].values.reshape(-1, 1).astype(float)
    prices_scaled = scaler.transform(prices)

    X_all, y_all = create_sliding_window(prices_scaled, window_size)
    dates_all = df["Date"].iloc[window_size:].reset_index(drop=True)
    X_all_3d = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

    return prices, prices_scaled, X_all_3d, y_all, dates_all


def forecast_future(model, prices_scaled, scaler, horizon, window_size=WINDOW_SIZE):
    """
    Forecast autoregressive:
    - Ambil window terakhir dari data scaled
    - Prediksi 1 hari, append ke window, geser, ulang hingga horizon
    """
    last_window = prices_scaled[-window_size:, 0].copy()
    preds_scaled = []

    for _ in range(horizon):
        x_input = last_window.reshape(1, window_size, 1)
        pred_scaled = model.predict(x_input, verbose=0)
        preds_scaled.append(pred_scaled[0, 0])
        last_window = np.append(last_window[1:], pred_scaled[0, 0])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds_scaled).flatten()
    return preds_inv


# ==============================
# STREAMLIT APP
# ==============================
def main():
    st.set_page_config(
        page_title="Prediksi BNB - LSTM & Bi-LSTM (Daily)",
        layout="wide",
    )

    # Global font: Poppins
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # HEADER
    header_html = f"""
    <div style="text-align:center; margin-bottom: 1.5rem;">
      <img src="{BNB_LOGO_URL}" alt="BNB Logo" width="120" style="margin-bottom:0.5rem;" />
      <h2 style="margin-bottom:0.25rem;">📈 Prediksi Harian Binance Coin (BNB) - LSTM &amp; Bi-LSTM</h2>
      <p style="font-size:0.9rem; color:#888;">
    Dashboard ini merupakan alat bantu analisis teknis dan peramalan data historis BNB yang diambil dari Yahoo Finance. Hasil prediksi bersifat edukatif dan eksperimental, bukan merupakan saran atau rekomendasi investasi.
      </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # SIDEBAR
    with st.sidebar:
        st.header("🔧 Pengaturan")
        horizon = st.number_input(
            "Horizon prediksi ke depan (hari)",
            min_value=1,
            max_value=30,
            value=DEFAULT_FORECAST_DAYS,
            step=1,
        )

    # LOAD DATA HARIAN
    try:
        df = load_daily_data()
    except Exception as e:
        st.error(f"Error saat mengambil data BNB harian: {e}")
        return

    if len(df) < WINDOW_SIZE + 2:
        st.error("Data terlalu sedikit untuk membuat window dan membedakan hari ini & kemarin.")
        return

    df = df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

    # =========================
    # HARI INI & KEMARIN (BERDASARKAN POSISI INDEX)
    # =========================
    today_date = df["Date"].iloc[-1].date()
    today_price = float(df["Price"].iloc[-1])

    yesterday_date = df["Date"].iloc[-2].date()
    yesterday_price = float(df["Price"].iloc[-2])

    # =========================
    # LOAD MODEL & SCALER
    # =========================
    try:
        lstm_model, bilstm_model, scaler = load_models_and_scaler()
    except Exception as e:
        st.error(f"Error saat memuat model/scaler: {e}")
        return

    # DATA UNTUK MODEL
    prices, prices_scaled, X_all_3d, y_all_scaled, dates_all = prepare_data(df, scaler)

    y_pred_lstm_scaled = lstm_model.predict(X_all_3d, verbose=0)
    y_pred_bilstm_scaled = bilstm_model.predict(X_all_3d, verbose=0)

    y_true_inv = scaler.inverse_transform(y_all_scaled.reshape(-1, 1)).flatten().astype(float)
    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm_scaled).flatten().astype(float)
    y_pred_bilstm_inv = scaler.inverse_transform(y_pred_bilstm_scaled).flatten().astype(float)

    # =========================
    # INDEX POSISI (BUKAN KALENDER)
    # =========================
    n_points = len(dates_all)
    if n_points < 2:
        st.error("Titik data untuk prediksi terlalu sedikit.")
        return

    idx_today = n_points - 1
    idx_yest = n_points - 2

    # Prediksi hari ini & kemarin (berdasarkan posisi)
    pred_today_lstm = float(y_pred_lstm_inv[idx_today])
    pred_today_bilstm = float(y_pred_bilstm_inv[idx_today])

    pred_yesterday_lstm = float(y_pred_lstm_inv[idx_yest])
    pred_yesterday_bilstm = float(y_pred_bilstm_inv[idx_yest])

    # Error hari ini
    err_today_abs_lstm = abs(pred_today_lstm - today_price)
    err_today_pct_lstm = (err_today_abs_lstm / today_price * 100.0) if today_price != 0 else np.nan

    err_today_abs_bi = abs(pred_today_bilstm - today_price)
    err_today_pct_bi = (err_today_abs_bi / today_price * 100.0) if today_price != 0 else np.nan

    # =========================
    # AKURASI 7 HARI KE BELAKANG (SAMPAI H-1, POSISI)
    # =========================
    end_idx = n_points - 1          # index hari ini
    start_idx = max(0, end_idx - EVAL_DAYS)  # mulai dari H-7
    idx_sel = np.arange(start_idx, end_idx)  # sampai H-1 (tidak termasuk hari ini)

    tanggal_last = dates_all.iloc[idx_sel]
    y_true_eval = y_true_inv[idx_sel]
    y_pred_lstm_eval = y_pred_lstm_inv[idx_sel]
    y_pred_bilstm_eval = y_pred_bilstm_inv[idx_sel]

    abs_pct_err_lstm = np.abs((y_true_eval - y_pred_lstm_eval) / y_true_eval) * 100
    abs_pct_err_bilstm = np.abs((y_true_eval - y_pred_bilstm_eval) / y_true_eval) * 100

    # =========================
    # FORECAST KE DEPAN
    # =========================
    future_lstm = forecast_future(
        lstm_model, prices_scaled, scaler, horizon=horizon, window_size=WINDOW_SIZE
    )
    future_bilstm = forecast_future(
        bilstm_model, prices_scaled, scaler, horizon=horizon, window_size=WINDOW_SIZE
    )

    last_daily_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_daily_date + dt.timedelta(days=1), periods=horizon, freq="D")

    tomorrow_date = future_dates[0].date()
    tomorrow_lstm = float(future_lstm[0])
    tomorrow_bilstm = float(future_bilstm[0])

    # =============================
    # RINGKASAN 4 KOLOM
    # =============================
    st.markdown("### 🧾 Ringkasan Harga & Prediksi (Daily)")

    col1, col2, col3, col4 = st.columns(4)

    # 1) Harga hari ini
    with col1:
        st.markdown(f"**{format_tanggal_indo(today_date)}**")
        st.markdown("Harga Hari Ini (Daily)")
        st.markdown(f"### ${today_price:,.4f}")

    # 2) Prediksi hari ini + error
    with col2:
        st.markdown(f"**{format_tanggal_indo(today_date)}**")
        st.markdown("Prediksi Hari Ini")
        st.markdown(f"**LSTM   :** ${pred_today_lstm:,.4f}")
        st.markdown(
            f"<span style='font-size:0.85rem;'>"
            f"Error LSTM vs data hari ini: {err_today_abs_lstm:,.4f} USD "
            f"({err_today_pct_lstm:,.2f}%)</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Bi-LSTM:** ${pred_today_bilstm:,.4f}")
        st.markdown(
            f"<span style='font-size:0.85rem;'>"
            f"Error Bi-LSTM vs data hari ini: {err_today_abs_bi:,.4f} USD "
            f"({err_today_pct_bi:,.2f}%)</span>",
            unsafe_allow_html=True,
        )

    # 3) Harga kemarin (data & prediksi) — pasti H-1
    with col3:
        st.markdown(f"**{format_tanggal_indo(yesterday_date)}**")
        st.markdown("Harga Kemarin (Aktual & Prediksi)")
        st.markdown(f"**Aktual   :** ${yesterday_price:,.4f}")
        st.markdown(f"**LSTM     :** ${pred_yesterday_lstm:,.4f}")
        st.markdown(f"**Bi-LSTM  :** ${pred_yesterday_bilstm:,.4f}")

    # 4) Prediksi besok
    with col4:
        st.markdown(f"**{format_tanggal_indo(tomorrow_date)}**")
        st.markdown("Prediksi Besok")
        st.markdown(f"**LSTM   :** ${tomorrow_lstm:,.4f}")
        st.markdown(f"**Bi-LSTM:** ${tomorrow_bilstm:,.4f}")

    # =============================
    # AKURASI 7 HARI KE BELAKANG
    # =============================
    st.markdown("### 📉 Akurasi 7 Hari ke Belakang (Daily)")

    hari_last = [hari_indo_from_date(t) for t in tanggal_last]

    df_last = pd.DataFrame({
        "Hari": hari_last,
        "Tanggal": [t.date() for t in tanggal_last],
        "Aktual (USD)": y_true_eval,
        "Prediksi LSTM (USD)": y_pred_lstm_eval,
        "Error LSTM (%)": abs_pct_err_lstm,
        "Prediksi Bi-LSTM (USD)": y_pred_bilstm_eval,
        "Error Bi-LSTM (%)": abs_pct_err_bilstm,
    })

    st.dataframe(
        df_last.style.format({
            "Aktual (USD)": "{:,.4f}",
            "Prediksi LSTM (USD)": "{:,.4f}",
            "Error LSTM (%)": "{:,.2f}",
            "Prediksi Bi-LSTM (USD)": "{:,.4f}",
            "Error Bi-LSTM (%)": "{:,.2f}",
        })
    )

    fig_past, ax_past = plt.subplots(figsize=(10, 4))
    ax_past.plot(tanggal_last, y_true_eval, label="Aktual (Daily)", linewidth=2)
    ax_past.plot(tanggal_last, y_pred_lstm_eval, label="Prediksi LSTM")
    ax_past.plot(tanggal_last, y_pred_bilstm_eval, label="Prediksi Bi-LSTM")
    ax_past.set_xlabel("Tanggal")
    ax_past.set_ylabel("Harga (USD)")
    ax_past.grid(True)
    ax_past.legend()
    st.pyplot(fig_past)

    # =============================
    # PREDIKSI HORIZON HARI KE DEPAN
    # =============================
    st.markdown(f"### 📈 Prediksi {horizon} Hari ke Depan (Daily)")

    hari_future = [hari_indo_from_date(t) for t in future_dates]

    df_future = pd.DataFrame({
        "Hari": hari_future,
        "Tanggal": future_dates.date,
        "Prediksi LSTM (USD)": future_lstm,
        "Prediksi Bi-LSTM (USD)": future_bilstm,
    })

    st.dataframe(
        df_future.style.format({
            "Prediksi LSTM (USD)": "{:,.4f}",
            "Prediksi Bi-LSTM (USD)": "{:,.4f}",
        })
    )

    fig_future, ax_future = plt.subplots(figsize=(10, 4))
    ax_future.plot(future_dates, future_lstm, marker="o", label="Forecast LSTM")
    ax_future.plot(future_dates, future_bilstm, marker="o", label="Forecast Bi-LSTM")
    ax_future.set_xlabel("Tanggal")
    ax_future.set_ylabel("Harga (USD)")
    ax_future.grid(True)
    ax_future.legend()
    st.pyplot(fig_future)


if __name__ == "__main__":
    main()
