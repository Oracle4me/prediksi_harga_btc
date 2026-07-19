import os
import sys
import json
import threading
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, Response
import joblib

sys.path.insert(0, os.path.dirname(__file__))

BASE = os.path.dirname(__file__)
SAVED = os.path.join(BASE, "models", "hasil_train")

app = Flask(__name__)


FORECAST_HORIZON = 1
LSTM_SEQ_LENGTH = 45
LSTM_EPOCHS = 120
BATCH_SIZE = 16

TUNE_RF_XGB = False
TUNE_LSTM = True


def safe_load_csv(name):
    path = os.path.join(SAVED, name)

    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except Exception:
        return pd.read_csv(path)


def safe_load_json(name):
    path = os.path.join(SAVED, name)

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return None


def clean_json_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def normalize_bool_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower == "true":
            return True

        if value_lower == "false":
            return False

        if value_lower in ["nan", "none", "null", "<na>"]:
            return None

    return bool(value)


def models_trained():
    needed = [
        "lstm_predictions.csv",
        "rf_predictions.csv",
        "xgb_predictions.csv",
        "simple_rnn_predictions.csv"
    ]

    return all(os.path.exists(os.path.join(SAVED, name)) for name in needed)


def clean_metric(metric):
    if metric is None:
        return None

    metric = dict(metric)

    if "best_params" in metric:
        metric["best_params"] = str(metric["best_params"])

    return metric


def add_trend_if_missing(df):
    if df is None or df.empty:
        return df

    df = df.copy()

    if "Actual_Trend" not in df.columns:
        df["Actual_Change"] = df["Actual"].diff()
        df["Actual_Trend"] = np.where(df["Actual_Change"] > 0, "Naik", "Turun")
        df.loc[df.index[0], "Actual_Trend"] = "-"

    if "Predicted_Trend" not in df.columns:
        df["Predicted_Change"] = df["Predicted"].diff()
        df["Predicted_Trend"] = np.where(df["Predicted_Change"] > 0, "Naik", "Turun")
        df.loc[df.index[0], "Predicted_Trend"] = "-"

    if "Trend_Correct" not in df.columns:
        df["Trend_Correct"] = (
            df["Actual_Trend"] == df["Predicted_Trend"]
        ).astype("boolean")
        df.loc[df.index[0], "Trend_Correct"] = pd.NA

    return df


def prediction_payload(df):
    if df is None:
        return None

    df = add_trend_if_missing(df)

    if len(df) > 300:
        step = max(1, len(df) // 300)
        df = df.iloc[::step]

    payload = {
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": [
            None if pd.isna(v) else round(float(v), 2)
            for v in df["Actual"].tolist()
        ],
        "predicted": [
            None if pd.isna(v) else round(float(v), 2)
            for v in df["Predicted"].tolist()
        ]
    }

    if "Actual_Trend" in df.columns:
        payload["actual_trend"] = [clean_json_value(v) for v in df["Actual_Trend"].tolist()]

    if "Predicted_Trend" in df.columns:
        payload["predicted_trend"] = [clean_json_value(v) for v in df["Predicted_Trend"].tolist()]

    if "Trend_Correct" in df.columns:
        payload["trend_correct"] = [normalize_bool_value(v) for v in df["Trend_Correct"].tolist()]

    return payload


def save_training_summary(metrics_list):
    os.makedirs(SAVED, exist_ok=True)

    with open(os.path.join(SAVED, "training_summary.json"), "w") as f:
        json.dump(metrics_list, f, indent=4)

    pd.DataFrame(metrics_list).to_csv(
        os.path.join(SAVED, "training_summary.csv"),
        index=False
    )


def get_best_metric(metrics_list):
    best = max(metrics_list, key=lambda x: x["R2"])
    return best, "R2"


@app.route("/")
def index():
    trained = models_trained()
    return render_template("index.html", trained=trained)


@app.route("/api/status")
def status():
    return jsonify({"trained": models_trained()})


@app.route("/api/metrics")
def metrics():
    lstm = safe_load_json("lstm_metrics.json")
    rf = safe_load_json("rf_metrics.json")
    xgb = safe_load_json("xgb_metrics.json")
    rnn = safe_load_json("simple_rnn_metrics.json")

    if not all([lstm, rf, xgb, rnn]):
        return jsonify({"error": "Model belum dilatih"}), 404

    metrics_list = [
        clean_metric(lstm),
        clean_metric(rnn),
        clean_metric(rf),
        clean_metric(xgb)
    ]

    best, criterion = get_best_metric(metrics_list)

    return jsonify({
        "LSTM": clean_metric(lstm),
        "SimpleRNN": clean_metric(rnn),
        "RandomForest": clean_metric(rf),
        "XGBoost": clean_metric(xgb),
        "best_model": best["model"],
        "best_criterion": criterion
    })


@app.route("/api/predictions/<model>")
def predictions(model):
    file_map = {
        "lstm": "lstm_predictions.csv",
        "rf": "rf_predictions.csv",
        "xgb": "xgb_predictions.csv",
        "rnn": "simple_rnn_predictions.csv"
    }

    if model not in file_map:
        return jsonify({"error": "Model tidak ditemukan"}), 404

    df = safe_load_csv(file_map[model])

    if df is None:
        return jsonify({"error": "Data prediksi belum tersedia"}), 404

    return jsonify(prediction_payload(df))


@app.route("/api/all_predictions")
def all_predictions():
    results = {}

    for key, filename in [
        ("lstm", "lstm_predictions.csv"),
        ("rf", "rf_predictions.csv"),
        ("xgb", "xgb_predictions.csv"),
        ("rnn", "simple_rnn_predictions.csv")
    ]:
        df = safe_load_csv(filename)

        if df is not None:
            results[key] = prediction_payload(df)

    return jsonify(results)


@app.route("/api/btc_history")
def btc_history():
    from utils.preprocessing import load_or_fetch_data

    df = load_or_fetch_data(start="2024-01-01")

    if df is None:
        return jsonify({"error": "Data tidak tersedia"}), 404

    df_2024 = df[df.index >= "2024-01-01"]
    df_weekly = df_2024["Close"].resample("W").last().dropna()

    return jsonify({
        "dates": df_weekly.index.strftime("%Y-%m-%d").tolist(),
        "prices": df_weekly.round(2).tolist()
    })


@app.route("/api/btc_forecast")
def btc_forecast():
    import tensorflow as tf
    from utils.preprocessing import load_or_fetch_data, add_technical_indicators, get_feature_columns

    model_path = os.path.join(SAVED, "lstm_model.keras")
    scaler_path = os.path.join(SAVED, "lstm_scaler.pkl")
    close_path = os.path.join(SAVED, "lstm_close_scaler.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, close_path]):
        return jsonify({"error": "Model belum ditraining"}), 404

    df_raw = load_or_fetch_data(start="2024-01-01")

    if df_raw is None or df_raw.empty:
        return jsonify({"error": "Data tidak tersedia"}), 404

    try:
        feature_cols = get_feature_columns()

        scaler = joblib.load(scaler_path)
        close_scaler = joblib.load(close_path)
        model = tf.keras.models.load_model(model_path)

        model_seq_len = model.input_shape[1]
        model_n_features = model.input_shape[2]

        seq_len = int(model_seq_len) if model_seq_len is not None else LSTM_SEQ_LENGTH

        if model_n_features != len(feature_cols):
            return jsonify({
                "error": "Jumlah fitur model tidak cocok dengan preprocessing.py terbaru.",
                "model_features": int(model_n_features),
                "current_features": len(feature_cols),
                "solution": "Hapus folder models/hasil_train atau jalankan ulang python train_all.py."
            }), 500

        df_ind = add_technical_indicators(df_raw.copy())

        if len(df_ind) < seq_len:
            return jsonify({"error": "Data historis tidak cukup untuk sequence LSTM."}), 500

        latest_features = df_ind[feature_cols].tail(seq_len).values
        latest_scaled = scaler.transform(latest_features)

        x_input = latest_scaled.reshape(1, seq_len, len(feature_cols))

        pred_scaled = model.predict(x_input, verbose=0)[0][0]

        pred_price = close_scaler.inverse_transform(
            np.array([[pred_scaled]])
        )[0][0]

        pred_price = float(pred_price)

        last_actual_date = df_raw.index[-1]
        last_actual_price = float(df_raw["Close"].iloc[-1])

        future_date = last_actual_date + pd.Timedelta(days=1)
        change_pct = ((pred_price - last_actual_price) / last_actual_price) * 100

        ci_upper = pred_price * 1.04
        ci_lower = pred_price * 0.96

        if change_pct >= 2.5:
            trend = "Strong Bullish"
            trend_color = "up"
            trend_icon = "bi-rocket-takeoff-fill"
            note = f"Model memprediksi kenaikan kuat sebesar {change_pct:.2f}% untuk hari berikutnya."

        elif change_pct >= 0.5:
            trend = "Bullish"
            trend_color = "up"
            trend_icon = "bi-arrow-up-circle-fill"
            note = f"Model memprediksi kenaikan sebesar {change_pct:.2f}% untuk hari berikutnya."

        elif change_pct <= -2.5:
            trend = "Strong Bearish"
            trend_color = "down"
            trend_icon = "bi-exclamation-octagon-fill"
            note = f"Model memprediksi penurunan kuat sebesar {change_pct:.2f}% untuk hari berikutnya."

        elif change_pct <= -0.5:
            trend = "Bearish"
            trend_color = "down"
            trend_icon = "bi-arrow-down-circle-fill"
            note = f"Model memprediksi penurunan sebesar {change_pct:.2f}% untuk hari berikutnya."

        else:
            trend = "Sideways"
            trend_color = "neutral"
            trend_icon = "bi-dash-circle-fill"
            note = f"Model memprediksi perubahan harga sebesar {change_pct:.2f}% untuk hari berikutnya."

        return jsonify({
            "dates": [future_date.strftime("%Y-%m-%d")],
            "forecast": [round(float(pred_price), 2)],
            "ci_upper": [round(float(ci_upper), 2)],
            "ci_lower": [round(float(ci_lower), 2)],
            "last_actual_date": last_actual_date.strftime("%Y-%m-%d"),
            "last_actual_price": round(last_actual_price, 2),
            "forecast_end": round(float(pred_price), 2),
            "change_pct": round(float(change_pct), 2),
            "trend": trend,
            "trend_color": trend_color,
            "trend_icon": trend_icon,
            "note": note,
            "method_note": "Forecast dilakukan untuk 1 hari ke depan menggunakan sequence historis terakhir oleh model LSTM."
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/idr_rate")
def idr_rate():
    import yfinance as yf

    try:
        ticker = yf.Ticker("IDR=X")
        hist = ticker.history(period="5d")

        if hist is None or hist.empty or "Close" not in hist.columns:
            return jsonify({
                "rate": None,
                "pair": "USD/IDR",
                "source": "yfinance",
                "error": "Data kurs USD/IDR tidak tersedia"
            }), 503

        close_series = hist["Close"].dropna()

        if close_series.empty:
            return jsonify({
                "rate": None,
                "pair": "USD/IDR",
                "source": "yfinance",
                "error": "Data kurs USD/IDR kosong"
            }), 503

        rate = float(close_series.iloc[-1])
        last_update = close_series.index[-1]

        if not np.isfinite(rate) or rate <= 0:
            return jsonify({
                "rate": None,
                "pair": "USD/IDR",
                "source": "yfinance",
                "error": "Nilai kurs USD/IDR tidak valid"
            }), 503

        return jsonify({
            "rate": round(rate, 2),
            "pair": "USD/IDR",
            "source": "yfinance",
            "last_update": str(last_update)
        })

    except Exception as e:
        return jsonify({
            "rate": None,
            "pair": "USD/IDR",
            "source": "yfinance",
            "error": str(e)
        }), 503


@app.route("/api/lstm_history")
def lstm_training_history():
    path = os.path.join(SAVED, "lstm_history.csv")

    if not os.path.exists(path):
        return jsonify({"error": "History tidak tersedia"}), 404

    df = pd.read_csv(path)

    return jsonify({
        "epochs": list(range(1, len(df) + 1)),
        "loss": [
            None if pd.isna(v) else round(float(v), 6)
            for v in df["loss"].tolist()
        ] if "loss" in df.columns else [],
        "val_loss": [
            None if pd.isna(v) else round(float(v), 6)
            for v in df["val_loss"].tolist()
        ] if "val_loss" in df.columns else [],
        "mae": [
            None if pd.isna(v) else round(float(v), 6)
            for v in df["mae"].tolist()
        ] if "mae" in df.columns else [],
        "val_mae": [
            None if pd.isna(v) else round(float(v), 6)
            for v in df["val_mae"].tolist()
        ] if "val_mae" in df.columns else []
    })


training_log = []
training_done = threading.Event()
training_lock = threading.Lock()


def do_train():
    global training_log

    with training_lock:
        training_log = []
        training_done.clear()

    def log(msg):
        with training_lock:
            training_log.append(msg)

    try:
        from utils.preprocessing import load_or_fetch_data, prepare_compare_data, prepare_lstm_data
        from models.compare_models import train_random_forest, train_xgboost, train_simple_rnn
        from models.lstm_model import train_lstm

        os.makedirs(SAVED, exist_ok=True)

        log("Mengambil data Bitcoin dari CSV/Yahoo Finance mulai 2024...")

        df = load_or_fetch_data(start="2024-01-01")

        if df is None or df.empty:
            log("Gagal memuat data.")
            training_done.set()
            return

        log(f"Data dimuat: {len(df)} baris ({df.index[0].date()} – {df.index[-1].date()})")
        log(f"Forecast horizon: {FORECAST_HORIZON} hari ke depan")
        log(f"LSTM sequence length: {LSTM_SEQ_LENGTH} hari")

        log("\n1. Training Random Forest...")

        X_tr, X_te, y_tr, y_te, dates_te, _ = prepare_compare_data(
            df,
            forecast_horizon=FORECAST_HORIZON
        )

        _, rf_m, _ = train_random_forest(
            X_tr,
            X_te,
            y_tr,
            y_te,
            dates_te,
            tune=TUNE_RF_XGB,
            save_dir=SAVED
        )

        log(
            f"Random Forest selesai — "
            f"MAE: {rf_m['MAE']:,.0f} | "
            f"R²: {rf_m['R2']:.4f} | "
        )

        log("\n2. Training XGBoost...")

        _, xgb_m, _ = train_xgboost(
            X_tr,
            X_te,
            y_tr,
            y_te,
            dates_te,
            tune=TUNE_RF_XGB,
            save_dir=SAVED
        )

        log(
            f"XGBoost selesai — "
            f"MAE: {xgb_m['MAE']:,.0f} | "
            f"R²: {xgb_m['R2']:.4f} | "
        )

        log("\n3. Training Simple RNN...")

        _, rnn_m, _ = train_simple_rnn(
            X_tr,
            X_te,
            y_tr,
            y_te,
            dates_te,
            epochs=LSTM_EPOCHS,
            batch_size=BATCH_SIZE,
            save_dir=SAVED
        )

        log(
            f"Simple RNN selesai — "
            f"MAE: {rnn_m['MAE']:,.0f} | "
            f"R²: {rnn_m['R2']:.4f} | "
        )

        log("\n4. Training LSTM Price Trend Boost...")

        (
            X_trl,
            X_tel,
            y_trl,
            y_tel,
            scaler,
            close_scaler,
            dates_tel,
            _
        ) = prepare_lstm_data(
            df,
            seq_length=LSTM_SEQ_LENGTH,
            forecast_horizon=FORECAST_HORIZON
        )

        _, lstm_m, _, _ = train_lstm(
            X_trl,
            X_tel,
            y_trl,
            y_tel,
            close_scaler,
            dates_tel,
            epochs=LSTM_EPOCHS,
            batch_size=BATCH_SIZE,
            save_dir=SAVED,
            tune=TUNE_LSTM
        )

        log(
            f"LSTM selesai — "
            f"MAE: {lstm_m['MAE']:,.0f} | "
            f"R²: {lstm_m['R2']:.4f} | "
        )

        joblib.dump(scaler, os.path.join(SAVED, "lstm_scaler.pkl"))
        joblib.dump(close_scaler, os.path.join(SAVED, "lstm_close_scaler.pkl"))

        metrics_list = [lstm_m, rnn_m, rf_m, xgb_m]
        save_training_summary(metrics_list)

        best, criterion = get_best_metric(metrics_list)

        log("\nRINGKASAN PERBANDINGAN:")
        log("Model terbaik ditentukan berdasarkan nilai R² tertinggi.")

        sorted_metrics = sorted(
            metrics_list,
            key=lambda x: x["R2"],
            reverse=True
        )

        for m in sorted_metrics:
            star = " ← TERBAIK" if m["model"] == best["model"] else ""

            log(
                f"  {m['model']:<15} "
                f"MAE={m['MAE']:>10,.0f}  "
                f"RMSE={m['RMSE']:>10,.0f}  "
                f"R²={m['R2']:.4f}  "
                f"MAPE={m['MAPE']:.2f}%  "
                f"{star}"
            )

        log("\nTRAINING SELESAI — Refresh halaman untuk melihat hasil.")

    except Exception as e:
        log(f"Error: {str(e)}")
        log(traceback.format_exc())

    finally:
        training_done.set()

@app.route("/api/train", methods=["POST"])
def start_training():
    if not training_done.is_set():
        return jsonify({"status": "already_running"})

    training_done.clear()

    thread = threading.Thread(target=do_train, daemon=True)
    thread.start()

    return jsonify({"status": "started"})

@app.route("/api/train_log")
def train_log_stream():
    def generate():
        last_idx = 0
        import time

        while not training_done.is_set() or last_idx < len(training_log):
            with training_lock:
                current = training_log[last_idx:]
                last_idx = len(training_log)

            for line in current:
                yield f"data: {json.dumps(line)}\n\n"

            time.sleep(0.5)

        yield "data: {\"done\": true}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/live_price")
def live_price():
    import yfinance as yf

    try:
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="1d", interval="1m")

        if hist.empty:
            return jsonify({"error": "Harga live tidak tersedia"}), 404

        price = hist["Close"].iloc[-1]

        return jsonify({
            "price": float(price)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Predikis Harga BTC — LSTM | Simple RNN | XGBoost | Random Forest")
    print("  Buka browser: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    training_done.set()

    app.run(
        debug=True,
        threaded=True,
        port=5000
    )
