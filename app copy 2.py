
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


@st.cache_data(show_spinner=False, ttl=60)  # refresh setiap 60 detik
def load_realtime_data(ticker=TICKER):
    """
    Ambil data BNB dari 1 Januari 2020 sampai sekarang (harian),
    lalu sinkronkan harga terakhir dengan data intraday (1 menit)
    agar mendekati harga realtime di Yahoo Finance.
    Output: DataFrame dengan kolom: Date, Price.
    """
    # 1. Data harian historis (untuk training & plotting)
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

    if daily.empty:
        raise RuntimeError("Data harian dari Yahoo Finance kosong. Coba ulang beberapa saat lagi.")

    daily = daily.reset_index()  # kolom 'Date' jadi kolom biasa
    if "Close" not in daily.columns:
        raise RuntimeError("Kolom 'Close' tidak ditemukan pada data Yahoo Finance (harian).")

    df = daily[["Date", "Close"]].rename(columns={"Close": "Price"})
    df = df.dropna().reset_index(drop=True)
    df["Price"] = df["Price"].astype(float)

    # 2. Data intraday (1 menit terakhir) untuk harga realtime
    try:
        ticker_yf = yf.Ticker(ticker)
        intraday = ticker_yf.history(period="1d", interval="1m")
    except Exception as e:
        # Kalau intraday error, kita pakai data harian saja
        intraday = pd.DataFrame()
        print(f"Warning: Gagal mengambil data intraday: {e}")

    if not intraday.empty:
        # Ambil harga terakhir (close bar 1 menit terakhir)
        last_ts = intraday.index[-1]
        last_price = float(intraday["Close"].iloc[-1])

        # Tanggal terakhir di data harian
        last_daily_date = df["Date"].iloc[-1]

        # 3. Sinkronisasi
        if last_ts.date() == last_daily_date.date():
            # Jika masih di tanggal yang sama → update baris terakhir dengan harga intraday
            df.loc[df.index[-1], "Price"] = last_price
        elif last_ts.date() > last_daily_date.date():
            # Jika intraday sudah masuk tanggal baru → tambahkan baris baru
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Date": [last_ts.normalize()],  # jam diset 00:00
                            "Price": [last_price],
                        }
                    ),
                ],
                ignore_index=True,
            )

    return df


def load_model_no_compile(path):
    """
    Load model Keras/TensorFlow tanpa compile
    untuk menghindari error:
    'Could not deserialize keras.metrics.mse ...'
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


def compute_last_n_metrics(y_true_inv, y_pred_inv, dates_all, n=EVAL_DAYS):
    """Hitung metrik dan error harian untuk N hari terakhir."""
    n = min(n, len(y_true_inv))
    y_true_last = y_true_inv[-n:]
    y_pred_last = y_pred_inv[-n:]
    dates_last = dates_all[-n:]

    mae = float(mean_absolute_error(y_true_last, y_pred_last))
    mape = float(mean_absolute_percentage_error(y_true_last, y_pred_last))
    abs_pct_err = np.abs((y_true_last - y_pred_last) / y_true_last) * 100

    return {
        "dates": dates_last,
        "y_true": y_true_last,
        "y_pred": y_pred_last,
        "mae": mae,
        "mape": mape,
        "abs_pct_err": abs_pct_err,
    }


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
        page_title="Prediksi Realtime BNB - Bi-LSTM",
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

    # HEADER CENTER: LOGO + JUDUL
    header_html = f"""
    <div style="text-align:center; margin-bottom: 1.5rem;">
      <img src="{BNB_LOGO_URL}" alt="BNB Logo" width="120" style="margin-bottom:0.5rem;" />
      <h2 style="margin-bottom:0.25rem;">📈 Prediksi Realtime Binance Coin (BNB) Bi-LSTM</h2>
      <p style="font-size:0.9rem; color:#888;">
        Dashboard ini menampilkan harga realtime BNB, prediksi LSTM &amp; Bi-LSTM, 
        akurasi 7 hari ke belakang, dan prediksi ke depan berdasarkan horizon yang kamu pilih.
      </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # SIDEBAR: HORIZON
    with st.sidebar:
        st.header("🔧 Pengaturan")
        horizon = st.number_input(
            "Horizon prediksi ke depan (hari)",
            min_value=1,
            max_value=30,
            value=DEFAULT_FORECAST_DAYS,
            step=1,
        )

    # LOAD DATA REALTIME
    try:
        df = load_realtime_data()
    except Exception as e:
        st.error(f"Error saat mengambil data BNB realtime: {e}")
        return

    if len(df) <= WINDOW_SIZE + EVAL_DAYS:
        st.error(
            f"Data terlalu sedikit (len={len(df)}). "
            f"Butuh minimal WINDOW_SIZE ({WINDOW_SIZE}) + EVAL_DAYS ({EVAL_DAYS})."
        )
        return

    # Harga terkini & kemarin
    today_price = float(df["Price"].iloc[-1])
    today_date = df["Date"].iloc[-1].date()

    yesterday_price = float(df["Price"].iloc[-2])
    yesterday_date = df["Date"].iloc[-2].date()

    # LOAD MODEL & SCALER
    try:
        lstm_model, bilstm_model, scaler = load_models_and_scaler()
    except Exception as e:
        st.error(f"Error saat memuat model/scaler: {e}")
        return

    # SIAPKAN DATA UNTUK MODEL
    prices, prices_scaled, X_all_3d, y_all_scaled, dates_all = prepare_data(df, scaler)

    y_pred_lstm_scaled = lstm_model.predict(X_all_3d, verbose=0)
    y_pred_bilstm_scaled = bilstm_model.predict(X_all_3d, verbose=0)

    y_true_inv = scaler.inverse_transform(y_all_scaled.reshape(-1, 1)).flatten().astype(float)
    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm_scaled).flatten().astype(float)
    y_pred_bilstm_inv = scaler.inverse_transform(y_pred_bilstm_scaled).flatten().astype(float)

    # EVALUASI 7 HARI TERAKHIR
    metrics_lstm = compute_last_n_metrics(y_true_inv, y_pred_lstm_inv, dates_all, n=EVAL_DAYS)
    metrics_bilstm = compute_last_n_metrics(y_true_inv, y_pred_bilstm_inv, dates_all, n=EVAL_DAYS)

    # Prediksi hari ini & kemarin dari LSTM & Bi-LSTM
    pred_today_lstm = float(y_pred_lstm_inv[-1])
    pred_today_bilstm = float(y_pred_bilstm_inv[-1])

    pred_yesterday_lstm = float(y_pred_lstm_inv[-2])
    pred_yesterday_bilstm = float(y_pred_bilstm_inv[-2])

    # Error hari ini (LSTM & Bi-LSTM) vs realtime
    err_today_abs_lstm = abs(pred_today_lstm - today_price)
    err_today_pct_lstm = (err_today_abs_lstm / today_price * 100.0) if today_price != 0 else np.nan

    err_today_abs_bi = abs(pred_today_bilstm - today_price)
    err_today_pct_bi = (err_today_abs_bi / today_price * 100.0) if today_price != 0 else np.nan

    # FORECAST KE DEPAN (horizon hari)
    future_lstm = forecast_future(
        lstm_model, prices_scaled, scaler, horizon=horizon, window_size=WINDOW_SIZE
    )
    future_bilstm = forecast_future(
        bilstm_model, prices_scaled, scaler, horizon=horizon, window_size=WINDOW_SIZE
    )

    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + dt.timedelta(days=1), periods=horizon, freq="D")

    tomorrow_date = future_dates[0].date()
    tomorrow_lstm = float(future_lstm[0])
    tomorrow_bilstm = float(future_bilstm[0])

    # =============================
    # RINGKASAN 4 KOLOM
    # =============================
    st.markdown("### 🧾 Ringkasan Harga & Prediksi")

    col1, col2, col3, col4 = st.columns(4)

    # 1) Harga realtime sekarang
    with col1:
        st.markdown(f"**{format_tanggal_indo(today_date)}**")
        st.markdown("Harga Realtime Sekarang")
        st.markdown(f"### ${today_price:,.4f}")

    # 2) Prediksi hari ini + error LSTM & Bi-LSTM
    with col2:
        st.markdown(f"**{format_tanggal_indo(today_date)}**")
        st.markdown("Prediksi Hari Ini")
        st.markdown(f"**LSTM   :** ${pred_today_lstm:,.4f}")
        st.markdown(
            f"<span style='color:green; font-size:0.85rem;'>"
            f"Error LSTM vs realtime: {err_today_abs_lstm:,.4f} USD "
            f"({err_today_pct_lstm:,.2f}%)</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Bi-LSTM:** ${pred_today_bilstm:,.4f}")
        st.markdown(
            f"<span style='color:green; font-size:0.85rem;'>"
            f"Error Bi-LSTM vs realtime: {err_today_abs_bi:,.4f} USD "
            f"({err_today_pct_bi:,.2f}%)</span>",
            unsafe_allow_html=True,
        )

    # 3) Harga kemarin (LSTM & Bi-LSTM)
    with col3:
        st.markdown(f"**{format_tanggal_indo(yesterday_date)}**")
        st.markdown("Harga Kemarin (LSTM & Bi-LSTM)")
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
    st.markdown("### 📉 Akurasi 7 Hari ke Belakang (Aktual vs LSTM vs Bi-LSTM)")

    tanggal_last = metrics_lstm["dates"]
    hari_last = [hari_indo_from_date(t) for t in tanggal_last]

    df_last = pd.DataFrame({
        "Hari": hari_last,
        "Tanggal": [t.date() for t in tanggal_last],
        "Aktual (USD)": metrics_lstm["y_true"],
        "Prediksi LSTM (USD)": metrics_lstm["y_pred"],
        "Error LSTM (%)": metrics_lstm["abs_pct_err"],
        "Prediksi Bi-LSTM (USD)": metrics_bilstm["y_pred"],
        "Error Bi-LSTM (%)": metrics_bilstm["abs_pct_err"],
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
    ax_past.plot(tanggal_last, metrics_lstm["y_true"], label="Aktual", linewidth=2)
    ax_past.plot(tanggal_last, metrics_lstm["y_pred"], label="Prediksi LSTM")
    ax_past.plot(tanggal_last, metrics_bilstm["y_pred"], label="Prediksi Bi-LSTM")
    ax_past.set_xlabel("Tanggal")
    ax_past.set_ylabel("Harga (USD)")
    ax_past.grid(True)
    ax_past.legend()
    st.pyplot(fig_past)

    # =============================
    # PREDIKSI HORIZON HARI KE DEPAN
    # =============================
    st.markdown(f"### 📈 Prediksi {horizon} Hari ke Depan")

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

