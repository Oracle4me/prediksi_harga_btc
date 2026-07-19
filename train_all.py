import os
import sys
import json
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from utils.preprocessing import load_or_fetch_data, prepare_compare_data, prepare_lstm_data
from models.compare_models import train_random_forest, train_xgboost, train_simple_rnn
from models.lstm_model import train_lstm  

BASE_DIR = os.path.dirname(__file__)
SAVE_DIR = os.path.join(BASE_DIR, "models", "hasil_train")
IMAGE_DIR = os.path.join(BASE_DIR, "static", "image")
os.makedirs(IMAGE_DIR, exist_ok=True)

START_DATE = "2024-01-01"

TEST_SIZE = 0.2
FORECAST_HORIZON = 1
LSTM_SEQ_LENGTH = 20  
EPOCHS = 80  

def print_metrics(metrics):
    mae = metrics.get("MAE", 0)
    rmse = metrics.get("RMSE", 0)
    r2 = metrics.get("R2", 0)
    mape = metrics.get("MAPE", 0)

    text = (
        f"MAE: {mae:,.2f} | "
        f"RMSE: {rmse:,.2f} | "
        f"R²: {r2:.4f} | "
        f"MAPE: {mape:.2f}%"
    )

    print("      " + text)

def save_summary(metrics_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "training_summary.json"), "w") as f:
        json.dump(metrics_list, f, indent=4)

    pd.DataFrame(metrics_list).to_csv(
        os.path.join(save_dir, "training_summary.csv"),
        index=False
    )

def get_best_model(metrics_list):
    sorted_metrics = sorted(
        metrics_list,
        key=lambda x: (-x["R2"], x["MAPE"])
    )

    best = sorted_metrics[0]
    return best, sorted_metrics


def main():
    print("=" * 70)
    print("  Prediksi Harga Bitcoin  — LSTM")
    print("  Perbandingan Model: LSTM | Simple RNN | XGBoost | Random Forest")
    print("=" * 70)

    print("\nKonfigurasi Eksperimen:")
    print(f"  Periode Data       : {START_DATE} – {'Terbaru' if None is None else None}")
    print(f"  Target             : Prediksi harga BTC Metode Regresi")
    print(f"  Forecast Horizon   : {FORECAST_HORIZON} hari ke depan")
    print(f"  Test Size          : {TEST_SIZE * 100:.0f}%")
    print(f"  LSTM Sequence      : {LSTM_SEQ_LENGTH} hari (proven optimal)")
    print(f"  Epochs             : {EPOCHS} (mencegah overfitting)")
    print(f"  Batch Size         : 16")
    print(f"  Architecture       : LSTM Optimized (2 layers, LayerNorm, minimal regularization)")

    print("\n[1/5] Memuat data Bitcoin dari CSV/Yahoo Finance...")

    df = load_or_fetch_data(
        start=START_DATE,
        end=None,
        force_update=False
    )

    if df is None or df.empty:
        print("GAGAL: Data tidak dapat dimuat.")
        return

    print(f"      Data dimuat: {len(df)} baris | {df.index[0].date()} – {df.index[-1].date()}")

    print("\n[2/5] Menyiapkan data ML dan training Random Forest...")

    X_tr, X_te, y_tr, y_te, dates_te, feats = prepare_compare_data(
        df,
        test_size=TEST_SIZE,
        forecast_horizon=FORECAST_HORIZON
    )

    print(f"      Jumlah fitur ML : {len(feats)}")
    print(f"      Train size      : {len(X_tr)}")
    print(f"      Test size       : {len(X_te)}")

    _, rf_metrics, _ = train_random_forest(
        X_tr,
        X_te,
        y_tr,
        y_te,
        dates_te,
        tune=False,
        save_dir=SAVE_DIR
    )

    print("      Random Forest selesai.")
    print_metrics(rf_metrics)

    print("\n[3/5] Training XGBoost...")

    _, xgb_metrics, _ = train_xgboost(
        X_tr,
        X_te,
        y_tr,
        y_te,
        dates_te,
        tune=False,
        save_dir=SAVE_DIR
    )

    print("      XGBoost selesai.")
    print_metrics(xgb_metrics)

    print("\n[4/5] Training Simple RNN...")

    _, rnn_metrics, _ = train_simple_rnn(
        X_tr,
        X_te,
        y_tr,
        y_te,
        dates_te,
        save_dir=SAVE_DIR,
        epochs=EPOCHS,
        batch_size=16
    )

    print("      Simple RNN selesai.")
    print_metrics(rnn_metrics)

    print("\n[5/5] Menyiapkan sequence data dan training LSTM Optimized...")

    (
        X_tr_l,
        X_te_l,
        y_tr_l,
        y_te_l,
        scaler,
        close_scaler,
        dates_te_l,
        lstm_feats
    ) = prepare_lstm_data(
        df,
        seq_length=LSTM_SEQ_LENGTH,
        test_size=TEST_SIZE,
        forecast_horizon=FORECAST_HORIZON
    )

    print(f"      Jumlah fitur LSTM : {len(lstm_feats)}")
    print(f"      Train sequence    : {len(X_tr_l)}")
    print(f"      Test sequence     : {len(X_te_l)}")
    print(f"      Input shape       : {X_tr_l.shape[1:]}")

    _, lstm_metrics, _, _ = train_lstm(
        X_tr_l,
        X_te_l,
        y_tr_l,
        y_te_l,
        close_scaler,
        dates_te_l,
        epochs=EPOCHS,
        batch_size=16,
        save_dir=SAVE_DIR,
        tune=True  # Tuning 3 configs untuk menemukan hasil terbaik
    )

    print("      LSTM selesai.")
    print_metrics(lstm_metrics)

    os.makedirs(SAVE_DIR, exist_ok=True)

    joblib.dump(scaler, os.path.join(SAVE_DIR, "lstm_scaler.pkl"))
    joblib.dump(close_scaler, os.path.join(SAVE_DIR, "lstm_close_scaler.pkl"))

    all_metrics = [
        lstm_metrics,
        rnn_metrics,
        xgb_metrics,
        rf_metrics
    ]

    save_summary(all_metrics, SAVE_DIR)

    best_model, sorted_metrics = get_best_model(all_metrics)

    print("\n" + "=" * 70)
    print("  RINGKASAN HASIL")
    print("=" * 70)
    print("  Model terbaik ditentukan berdasarkan nilai R² tertinggi dan didukung oleh nilai MAPE yang lebih rendah.")
    print("-" * 70)

    for m in sorted_metrics:
        star = " ★ TERBAIK" if m["model"] == best_model["model"] else ""

        line = (
            f"  {m['model']:<15} "
            f"MAE={m['MAE']:>10,.2f}  "
            f"RMSE={m['RMSE']:>10,.2f}  "
            f"R²={m['R2']:.4f}  "
            f"MAPE={m['MAPE']:.2f}%  "
        )

        line += star
        print(line)

    print("=" * 70)

    print("\nModel dan hasil training tersimpan di:")
    print(f"  {SAVE_DIR}")

    print("\nFile output utama:")
    print("  - rf_model.pkl")
    print("  - xgb_model.pkl")
    print("  - simple_rnn_model.keras")
    print("  - lstm_model.keras")
    print("  - lstm_scaler.pkl")
    print("  - lstm_close_scaler.pkl")
    print("  - lstm_tuning_results.csv")
    print("  - training_summary.json")
    print("  - training_summary.csv")

    print("\nJalankan `python app.py` untuk membuka web dashboard.\n")


if __name__ == "__main__":
    main()