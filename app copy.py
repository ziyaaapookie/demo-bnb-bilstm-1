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
LOOKBACK_DAYS = 200       # ambil data historis 200 hari terakhir
EVAL_DAYS = 7             # 7 hari ke belakang untuk evaluasi
DEFAULT_HORIZON = 7       # 7 hari ke depan untuk forecast
TICKER = "BNB-USD"

ARTEFACT_DIR = "artefak_bnb"
LSTM_PATH = os.path.join(ARTEFACT_DIR, "lstm_bnb.h5")
BILSTM_PATH = os.path.join(ARTEFACT_DIR, "bilstm_bnb.h5")
SCALER_PATH = os.path.join(ARTEFACT_DIR, "scaler_bnb.pkl")


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


@st.cache_data(show_spinner=False)
def load_realtime_data(ticker=TICKER, lookback_days=LOOKBACK_DAYS):
    """
    Ambil data BNB realtime dari Yahoo Finance.
    Menghasilkan DataFrame dengan kolom: Date, Price.
    """
    try:
        data = yf.download(
            ticker,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"Gagal mengambil data dari Yahoo Finance: {e}")

    if data.empty:
        raise RuntimeError("Data dari Yahoo Finance kosong. Coba ulang beberapa saat lagi.")

    data = data.reset_index()  # kolom 'Date' jadi kolom biasa
    if "Close" not in data.columns:
        raise RuntimeError("Kolom 'Close' tidak ditemukan pada data Yahoo Finance.")

    df = data[["Date", "Close"]].rename(columns={"Close": "Price"})
    df = df.dropna().reset_index(drop=True)

    # pastikan tipe float
    df["Price"] = df["Price"].astype(float)

    return df


def load_model_no_compile(path):
    """
    Load model Keras/TensorFlow tanpa compile
    untuk menghindari error:
    'Could not deserialize keras.metrics.mse ...'
    """
    # 1) Coba lewat tf.keras (umum)
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e1:
        # 2) Fallback ke API keras baru (jika ada)
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


def forecast_future(model, prices_scaled, scaler, horizon=DEFAULT_HORIZON, window_size=WINDOW_SIZE):
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
        page_title="BNB Real-Time Prediction - LSTM vs Bi-LSTM",
        layout="wide",
    )

    st.title("📈 Prediksi Realtime Binance Coin (BNB) - LSTM vs Bi-LSTM")

    st.write(
        """
        Aplikasi ini:
        - Mengambil **data harga BNB realtime** dari Yahoo Finance (ticker `BNB-USD`)
        - Menampilkan **harga kemarin, hari ini, dan besok (prediksi)**
        - Menampilkan **aktual vs prediksi 7 hari ke belakang** (LSTM & Bi-LSTM) + error (%)
        - Menampilkan **forecast 7 hari ke depan** (LSTM & Bi-LSTM)
        """
    )

    with st.sidebar:
        st.header("⚙️ Pengaturan")
        horizon = st.number_input(
            "Horizon prediksi ke depan (hari)",
            min_value=1,
            max_value=30,
            value=DEFAULT_HORIZON,
            step=1,
        )
        st.caption(
            f"Data historis diambil {LOOKBACK_DAYS} hari ke belakang.\n"
            f"Evaluasi akurasi menggunakan {EVAL_DAYS} hari terakhir."
        )

    # ------------------------------
    # Load data realtime
    # ------------------------------
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

    # Info dasar harga (pastikan float, bukan Series)
    today_price = float(df["Price"].iloc[-1])
    today_date = df["Date"].iloc[-1].date()

    yesterday_price = float(df["Price"].iloc[-2])
    yesterday_date = df["Date"].iloc[-2].date()

    st.subheader("📊 Data Realtime BNB (Yahoo Finance)")
    st.dataframe(df.tail(10))

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric(
            label=f"Harga Hari Ini ({today_date})",
            value=f"${today_price:,.4f}",
        )
        st.metric(
            label=f"Harga Kemarin ({yesterday_date})",
            value=f"${yesterday_price:,.4f}",
            delta=f"{today_price - yesterday_price:,.4f} USD",
        )

    # ------------------------------
    # Load model & scaler
    # ------------------------------
    try:
        lstm_model, bilstm_model, scaler = load_models_and_scaler()
    except Exception as e:
        st.error(f"Error saat memuat model/scaler: {e}")
        return

    # ------------------------------
    # Siapkan data untuk model
    # ------------------------------
    prices, prices_scaled, X_all_3d, y_all_scaled, dates_all = prepare_data(df, scaler)

    # Prediksi seluruh window untuk evaluasi
    y_pred_lstm_scaled = lstm_model.predict(X_all_3d, verbose=0)
    y_pred_bilstm_scaled = bilstm_model.predict(X_all_3d, verbose=0)

    # Invers scaling (pastikan float)
    y_true_inv = scaler.inverse_transform(y_all_scaled.reshape(-1, 1)).flatten().astype(float)
    y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm_scaled).flatten().astype(float)
    y_pred_bilstm_inv = scaler.inverse_transform(y_pred_bilstm_scaled).flatten().astype(float)

    # ------------------------------
    # Evaluasi 7 hari terakhir
    # ------------------------------
    metrics_lstm = compute_last_n_metrics(y_true_inv, y_pred_lstm_inv, dates_all, n=EVAL_DAYS)
    metrics_bilstm = compute_last_n_metrics(y_true_inv, y_pred_bilstm_inv, dates_all, n=EVAL_DAYS)

    st.subheader(f"📐 Akurasi {EVAL_DAYS} Hari Terakhir (Aktual vs Prediksi)")

    col_lstm, col_bilstm = st.columns(2)
    with col_lstm:
        st.markdown("### LSTM")
        st.write(f"**MAE**  : {metrics_lstm['mae']:.4f} USD")
        st.write(f"**MAPE** : {metrics_lstm['mape']:.2f} %")
    with col_bilstm:
        st.markdown("### Bi-LSTM")
        st.write(f"**MAE**  : {metrics_bilstm['mae']:.4f} USD")
        st.write(f"**MAPE** : {metrics_bilstm['mape']:.2f} %")

    # Tabel detail 7 hari terakhir
    st.markdown(f"#### 📅 Detail {EVAL_DAYS} Hari Terakhir")

    df_last = pd.DataFrame({
        "Tanggal": metrics_lstm["dates"].dt.date,
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

    # Chart 7 hari terakhir
    st.markdown("#### 📉 Grafik Aktual vs Prediksi (7 Hari Terakhir)")
    fig_past, ax_past = plt.subplots(figsize=(10, 4))
    ax_past.plot(metrics_lstm["dates"], metrics_lstm["y_true"], label="Aktual", linewidth=2)
    ax_past.plot(metrics_lstm["dates"], metrics_lstm["y_pred"], label="Prediksi LSTM")
    ax_past.plot(metrics_bilstm["dates"], metrics_bilstm["y_pred"], label="Prediksi Bi-LSTM")
    ax_past.set_xlabel("Tanggal")
    ax_past.set_ylabel("Harga (USD)")
    ax_past.grid(True)
    ax_past.legend()
    st.pyplot(fig_past)

    # ------------------------------
    # Forecast ke depan (horizon hari)
    # ------------------------------
    st.subheader(f"🔮 Prediksi {horizon} Hari ke Depan")

    future_lstm = forecast_future(
        lstm_model, prices_scaled, scaler, horizon=horizon, window_size=WINDOW_SIZE
    )
    future_bilstm = forecast_future(
        bilstm_model, prices_scaled, scaler, horizon=horizon, window_size=WINDOW_SIZE
    )

    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + dt.timedelta(days=1), periods=horizon, freq="D")

    # Harga besok (prediksi)
    tomorrow_date = future_dates[0].date()
    tomorrow_lstm = float(future_lstm[0])
    tomorrow_bilstm = float(future_bilstm[0])

    col_tmr1, col_tmr2 = st.columns(2)
    with col_tmr1:
        st.metric(
            label=f"Prediksi Besok (LSTM) - {tomorrow_date}",
            value=f"${tomorrow_lstm:,.4f}",
        )
    with col_tmr2:
        st.metric(
            label=f"Prediksi Besok (Bi-LSTM) - {tomorrow_date}",
            value=f"${tomorrow_bilstm:,.4f}",
        )

    # Tabel forecast
    df_future = pd.DataFrame({
        "Tanggal": future_dates.date,
        "Prediksi LSTM (USD)": future_lstm,
        "Prediksi Bi-LSTM (USD)": future_bilstm,
    })

    st.markdown("#### 📅 Tabel Prediksi ke Depan")
    st.dataframe(
        df_future.style.format({
            "Prediksi LSTM (USD)": "{:,.4f}",
            "Prediksi Bi-LSTM (USD)": "{:,.4f}",
        })
    )

    # Chart forecast
    st.markdown("#### 📈 Grafik Prediksi ke Depan")
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
