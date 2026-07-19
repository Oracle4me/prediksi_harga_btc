import os
import json
import random

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Deklarasika seed sebanyak 42
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def forecast_direction(y_actual, y_pred):
    """
    Prediksi hari ini dibandingkan dengan harga aktual sebelumnya.
    """
    y_actual = np.array(y_actual).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_actual) < 2 or len(y_pred) < 2:
        return 0.0

    previous_actual = y_actual[:-1]
    actual_now = y_actual[1:]
    pred_now = y_pred[1:]

    actual_trend = np.where(actual_now > previous_actual, 1, 0)
    pred_trend = np.where(pred_now > previous_actual, 1, 0)

    return round(float(np.mean(actual_trend == pred_trend) * 100), 4)


def compute_metrics(y_actual, y_pred, name):
    y_actual = np.array(y_actual).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    y_actual_safe = np.where(y_actual == 0, np.nan, y_actual)
    mape = np.nanmean(np.abs((y_actual - y_pred) / y_actual_safe)) * 100

    return {
        "model": name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MSE": round(float(mse), 4),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 4),
    }


def add_columns(pred_df):
    pred_df = pred_df.copy()

    pred_df["Baseline"] = pred_df["Actual"].shift(1)
    pred_df.loc[pred_df.index[0], "Baseline"] = pred_df.loc[pred_df.index[0], "Actual"]

    pred_df["Actual_Change"] = pred_df["Actual"] - pred_df["Baseline"]
    pred_df["Predicted_Change"] = pred_df["Predicted"] - pred_df["Baseline"]

    pred_df["Actual_Trend"] = np.where(pred_df["Actual_Change"] > 0, "Naik", "Turun")
    pred_df["Predicted_Trend"] = np.where(pred_df["Predicted_Change"] > 0, "Naik", "Turun")

    pred_df.loc[pred_df.index[0], "Actual_Trend"] = "-"
    pred_df.loc[pred_df.index[0], "Predicted_Trend"] = "-"

    pred_df["Trend_Correct"] = (
        pred_df["Actual_Trend"] == pred_df["Predicted_Trend"]
    ).astype("boolean")

    pred_df.loc[pred_df.index[0], "Trend_Correct"] = pd.NA

    return pred_df

# Pembagian data untuk validasi
def split_train_validation(X_train, y_train, val_size=0.2):
    val_len = int(len(X_train) * val_size)

    if val_len < 10:
        raise ValueError("Data training terlalu sedikit untuk validation set.")

    X_train_final = X_train[:-val_len]
    y_train_final = y_train[:-val_len]

    X_val = X_train[-val_len:]
    y_val = y_train[-val_len:]

    return X_train_final, X_val, y_train_final, y_val

# Model LSTM
def build_lstm_model(input_shape, units_1=96, units_2=48, dropout=0.12, learning_rate=0.0007):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import Huber

    model = Sequential([
        Input(shape=input_shape),

        LSTM(units_1, return_sequences=True),
        LayerNormalization(), # Stabilisasi untuk mengurangi overfitting
        Dropout(dropout),

        LSTM(units_2, return_sequences=False),
        LayerNormalization(),
        Dropout(dropout),

        Dense(64, activation="relu"),
        Dropout(dropout / 2),

        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=Huber(delta=0.05),
        metrics=["mae"]
    )

    return model


def train_single_lstm_config(X_train, y_train, X_val, y_val, input_shape, config, epochs, verbose=1):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = build_lstm_model(
        input_shape=input_shape,
        units_1=config["units_1"],
        units_2=config["units_2"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"]
    )

    callbacks = [
        # Pencegahan Overfitting => Menghentikan training jika validation loss jika tidak membaik selama 8-10 epochs
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("patience", 10),
            restore_best_weights=True
        ),
        # Pencegahan Underfitting => Menghentikan training jika validation loss plateau jika tidak ada kemajuan
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=False,
        verbose=verbose
    )

    best_val_loss = min(history.history["val_loss"])

    return model, history, best_val_loss

def _build_directional_signals(y_actual_reference, y_pred):
    """
    Membuat beberapa sinyal arah dari data historis yang memang sudah diketahui
    pada skenario one-step forecasting.

    Semua sinyal memakai baseline harga aktual sebelumnya, bukan harga aktual hari ini.
    """
    y_actual_reference = np.array(y_actual_reference).flatten()
    y_pred = np.array(y_pred).flatten()

    prev = np.r_[y_actual_reference[0], y_actual_reference[:-1]]

    s = pd.Series(prev)

    prev_safe = np.where(prev == 0, np.nan, prev)

    raw = (y_pred - prev) / prev_safe

    mom1 = s.pct_change(1).fillna(0).values
    mom2 = s.pct_change(2).fillna(0).values
    mom3 = s.pct_change(3).fillna(0).values
    mom5 = s.pct_change(5).fillna(0).values
    mom7 = s.pct_change(7).fillna(0).values
    mom14 = s.pct_change(14).fillna(0).values

    ma3 = s.rolling(3, min_periods=1).mean()
    ma7 = s.rolling(7, min_periods=1).mean()
    ma14 = s.rolling(14, min_periods=1).mean()
    ma21 = s.rolling(21, min_periods=1).mean()

    ma3_7 = ((ma3 - ma7) / ma7.replace(0, np.nan)).fillna(0).values
    ma7_14 = ((ma7 - ma14) / ma14.replace(0, np.nan)).fillna(0).values
    ma14_21 = ((ma14 - ma21) / ma21.replace(0, np.nan)).fillna(0).values

    ma_slope7 = ma7.pct_change(3).fillna(0).values
    ma_slope14 = ma14.pct_change(5).fillna(0).values

    distance_ma7 = ((s - ma7) / ma7.replace(0, np.nan)).fillna(0).values
    distance_ma14 = ((s - ma14) / ma14.replace(0, np.nan)).fillna(0).values

    # Mean reversion: kalau terlalu jauh di atas MA, arah berikutnya cenderung koreksi.
    mean_reversion = -(0.65 * distance_ma7 + 0.35 * distance_ma14)

    volatility = s.pct_change().rolling(7, min_periods=1).std().fillna(0).values

    signals = {
        "raw": raw,
        "mom1": mom1,
        "mom2": mom2,
        "mom3": mom3,
        "mom5": mom5,
        "mom7": mom7,
        "mom14": mom14,
        "ma3_7": ma3_7,
        "ma7_14": ma7_14,
        "ma14_21": ma14_21,
        "ma_slope7": ma_slope7,
        "ma_slope14": ma_slope14,
        "mean_reversion": mean_reversion,
        "volatility": volatility
    }

    for key, value in signals.items():
        value = np.array(value, dtype=float)
        value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        signals[key] = value

    return prev, signals


def _strategy_score(signals, strategy):
    """
    Kumpulan strategi arah. Strategi terbaik dipilih dari validation set.
    """
    raw = signals["raw"]
    mom1 = signals["mom1"]
    mom2 = signals["mom2"]
    mom3 = signals["mom3"]
    mom5 = signals["mom5"]
    mom7 = signals["mom7"]
    mom14 = signals["mom14"]
    ma3_7 = signals["ma3_7"]
    ma7_14 = signals["ma7_14"]
    ma14_21 = signals["ma14_21"]
    ma_slope7 = signals["ma_slope7"]
    ma_slope14 = signals["ma_slope14"]
    mean_reversion = signals["mean_reversion"]

    if strategy == "raw":
        score = raw

    elif strategy == "momentum_short":
        score = 0.50 * mom1 + 0.30 * mom2 + 0.20 * mom3

    elif strategy == "momentum_mid":
        score = 0.20 * mom3 + 0.35 * mom5 + 0.45 * mom7

    elif strategy == "momentum_long":
        score = 0.25 * mom7 + 0.75 * mom14

    elif strategy == "ma_trend":
        score = 0.35 * ma3_7 + 0.40 * ma7_14 + 0.25 * ma_slope7

    elif strategy == "ma_long_trend":
        score = 0.30 * ma7_14 + 0.35 * ma14_21 + 0.35 * ma_slope14

    elif strategy == "raw_plus_short_momentum":
        score = 0.35 * raw + 0.40 * mom1 + 0.25 * mom3

    elif strategy == "raw_plus_mid_momentum":
        score = 0.30 * raw + 0.30 * mom3 + 0.40 * mom7

    elif strategy == "raw_plus_ma":
        score = 0.30 * raw + 0.35 * ma7_14 + 0.35 * ma_slope7

    elif strategy == "trend_stack":
        score = (
            0.20 * raw +
            0.20 * mom3 +
            0.25 * mom7 +
            0.20 * ma7_14 +
            0.15 * ma_slope7
        )

    elif strategy == "long_trend_stack":
        score = (
            0.15 * raw +
            0.20 * mom7 +
            0.25 * mom14 +
            0.20 * ma14_21 +
            0.20 * ma_slope14
        )

    elif strategy == "mean_reversion":
        score = 0.30 * raw + 0.70 * mean_reversion

    elif strategy == "contrarian_short":
        score = 0.20 * raw - 0.45 * mom1 - 0.35 * mom3

    elif strategy == "contrarian_mid":
        score = 0.20 * raw - 0.35 * mom3 - 0.45 * mom7

    else:
        score = raw

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def _direction_to_price(prev, direction, magnitude):
    """
    Mengubah keputusan arah menjadi harga prediksi.
    """
    prev = np.array(prev).flatten()
    direction = np.array(direction).astype(int).flatten()

    sign = np.where(direction == 1, 1.0, -1.0)
    return prev * (1.0 + sign * magnitude)


def _accuracy_from_direction(y_actual, direction):
    y_actual = np.array(y_actual).flatten()

    if len(y_actual) < 2:
        return 0.0

    prev = y_actual[:-1]
    actual_now = y_actual[1:]
    direction = np.array(direction).astype(int).flatten()[1:]

    actual_direction = np.where(actual_now > prev, 1, 0)

    return round(float(np.mean(actual_direction == direction) * 100), 4)


def optimize_boost(y_actual, y_pred):
    """
    Dia memilih strategi arah terbaik berdasarkan validation set:
    - momentum pendek
    - momentum menengah
    - MA trend
    - contrarian / mean-reversion
    """
    y_actual = np.array(y_actual).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_actual) < 30:
        return {
            "strategy": "raw",
            "threshold": 0.0,
            "magnitude": 0.01,
            "val_trend_accuracy": 0.0
        }

    prev, signals = _build_directional_signals(y_actual, y_pred)

    returns = pd.Series(y_actual).pct_change().abs().dropna()
    magnitude_candidates = [
        0.003,
        0.005,
        0.0075,
        0.010,
        0.0125,
        0.015,
        0.020
    ]

    if not returns.empty:
        med = float(returns.median())
        if np.isfinite(med) and med > 0:
            magnitude_candidates.extend([
                float(np.clip(med * 0.50, 0.003, 0.025)),
                float(np.clip(med * 0.75, 0.003, 0.025)),
                float(np.clip(med * 1.00, 0.003, 0.025)),
                float(np.clip(med * 1.25, 0.003, 0.025)),
            ])

    strategies = [
        "raw",
        "momentum_short",
        "momentum_mid",
        "momentum_long",
        "ma_trend",
        "ma_long_trend",
        "raw_plus_short_momentum",
        "raw_plus_mid_momentum",
        "raw_plus_ma",
        "trend_stack",
        "long_trend_stack",
        "mean_reversion",
        "contrarian_short",
        "contrarian_mid"
    ]

    thresholds = [
        -0.030,
        -0.020,
        -0.015,
        -0.010,
        -0.0075,
        -0.005,
        -0.0025,
        0.0,
        0.0025,
        0.005,
        0.0075,
        0.010,
        0.015,
        0.020,
        0.030
    ]

    best = {
        "strategy": "raw",
        "threshold": 0.0,
        "magnitude": 0.01,
        "val_trend_accuracy": -1.0
    }

    for strategy in strategies:
        score = _strategy_score(signals, strategy)

        for threshold in thresholds:
            direction = (score >= threshold).astype(int)
            acc = _accuracy_from_direction(y_actual, direction)

            if acc > best["val_trend_accuracy"]:
                best = {
                    "strategy": strategy,
                    "threshold": float(threshold),
                    "magnitude": 0.01,
                    "val_trend_accuracy": float(acc)
                }

    # Pilih magnitude yang menjaga harga tidak terlalu ngawur, setelah arah terbaik didapat.
    best_score = _strategy_score(signals, best["strategy"])
    best_direction = (best_score >= best["threshold"]).astype(int)

    best_magnitude = 0.01
    best_mae = float("inf")

    for magnitude in sorted(set(magnitude_candidates)):
        pred_price = _direction_to_price(prev, best_direction, magnitude)
        mae = mean_absolute_error(y_actual, pred_price)

        if mae < best_mae:
            best_mae = mae
            best_magnitude = float(magnitude)

    best["magnitude"] = best_magnitude
    best["val_mae_after_boost"] = round(float(best_mae), 4)

    return best


def apply_boost(y_actual_reference, y_pred, params):
    """
    Menerapkan directional boost.
    Baseline memakai harga aktual sebelumnya, yang memang diketahui saat one-step forecasting.
    """
    y_actual_reference = np.array(y_actual_reference).flatten()
    y_pred = np.array(y_pred).flatten()

    prev, signals = _build_directional_signals(y_actual_reference, y_pred)

    strategy = params.get("strategy", "raw")
    threshold = params.get("threshold", 0.0)
    magnitude = params.get("magnitude", 0.01)

    score = _strategy_score(signals, strategy)
    direction = (score >= threshold).astype(int)

    boosted_pred = _direction_to_price(prev, direction, magnitude)

    return boosted_pred


def train_lstm(

    X_train,
    X_test,
    y_train,
    y_test,
    close_scaler,
    dates_test,
    epochs=50,
    batch_size=16,
    save_dir="models",
    tune=True,
    val_size=0.2
):
    """
    LSTM price regression dengan trend boost.
    Target tetap harga, epoch dibatasi 50.
    """

    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)

    input_shape = (X_train.shape[1], X_train.shape[2])

    X_train_final, X_val, y_train_final, y_val = split_train_validation(
        X_train,
        y_train,
        val_size=val_size
    )

    if tune:
        configs = [
            {
                "name": "PriceBoostLSTM_64_32_do010_lr0007",
                "units_1": 64,
                "units_2": 32,
                "dropout": 0.10,
                "learning_rate": 0.0007,
                "batch_size": 16,
                "patience": 8
            },
            {
                "name": "PriceBoostLSTM_96_48_do012_lr0005",
                "units_1": 96,
                "units_2": 48,
                "dropout": 0.12,
                "learning_rate": 0.0005,
                "batch_size": 16,
                "patience": 9
            },
            {
                "name": "PriceBoostLSTM_128_64_do015_lr0003",
                "units_1": 128,
                "units_2": 64,
                "dropout": 0.15,
                "learning_rate": 0.0003,
                "batch_size": 16,
                "patience": 10
            }
        ]
    else:
        configs = [
            {
                "name": "PriceBoostLSTM_Default",
                "units_1": 96,
                "units_2": 48,
                "dropout": 0.12,
                "learning_rate": 0.0005,
                "batch_size": batch_size,
                "patience": 9
            }
        ]

    best_model = None
    best_history = None
    best_config = None
    best_val_loss = float("inf")
    best_val_trend_acc = -1.0
    best_boost_params = None

    tuning_results = []

    y_val_actual = close_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    for idx, config in enumerate(configs, start=1):
        print(f"\n      Training konfigurasi LSTM {idx}/{len(configs)}: {config['name']}")

        model, history, val_loss = train_single_lstm_config(
            X_train=X_train_final,
            y_train=y_train_final,
            X_val=X_val,
            y_val=y_val,
            input_shape=input_shape,
            config=config,
            epochs=epochs,
            verbose=1
        )

        val_pred_scaled = model.predict(X_val, verbose=0)
        val_pred_raw = close_scaler.inverse_transform(val_pred_scaled).flatten()

        raw_val_trend_acc = forecast_direction(y_val_actual, val_pred_raw)
        boost_params = optimize_boost(y_val_actual, val_pred_raw)
        val_pred_boosted = apply_boost(y_val_actual, val_pred_raw, boost_params)
        boosted_val_trend_acc = forecast_direction(y_val_actual, val_pred_boosted)

        tuning_results.append({
            "config": config["name"],
            "units_1": config["units_1"],
            "units_2": config["units_2"],
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "best_val_loss": round(float(val_loss), 6),
            "raw_val_trend_accuracy": round(float(raw_val_trend_acc), 4),
            "boosted_val_trend_accuracy": round(float(boosted_val_trend_acc), 4),

            # Versi boost baru memakai strategy/threshold/magnitude.
            # Key alpha/beta/clip dibuat optional agar tidak KeyError.
            "boost_strategy": boost_params.get("strategy"),
            "boost_threshold": boost_params.get("threshold"),
            "boost_magnitude": boost_params.get("magnitude"),
            "boost_alpha": boost_params.get("alpha"),
            "boost_beta": boost_params.get("beta"),
            "boost_clip": boost_params.get("clip"),
            "boost_val_mae_after": boost_params.get("val_mae_after_boost")
        })

        print(f"      Val Loss: {val_loss:.6f} | ")

        if best_model is None or boosted_val_trend_acc > best_val_trend_acc:
            best_val_loss = val_loss
            best_val_trend_acc = boosted_val_trend_acc
            best_model = model
            best_history = history
            best_config = config
            best_boost_params = boost_params

    y_pred_scaled = best_model.predict(X_test, verbose=0)

    y_pred_raw = close_scaler.inverse_transform(y_pred_scaled).flatten()
    y_actual_flat = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    y_pred_boosted = apply_boost(y_actual_flat, y_pred_raw, best_boost_params)

    metrics = compute_metrics(y_actual_flat, y_pred_boosted, "LSTM")
    metrics["best_config"] = best_config["name"]
    metrics["best_val_loss"] = round(float(best_val_loss), 6)
    metrics["epochs_limit"] = epochs

    best_model.save(os.path.join(save_dir, "lstm_model.keras"))

    pred_df = pd.DataFrame({
        "Date": dates_test,
        "Actual": y_actual_flat,
        "Predicted": y_pred_boosted,
        "Raw_Predicted": y_pred_raw
    })

    pred_df = add_columns(pred_df)
    pred_df.to_csv(os.path.join(save_dir, "lstm_predictions.csv"), index=False)

    hist_df = pd.DataFrame(best_history.history)
    hist_df.to_csv(os.path.join(save_dir, "lstm_history.csv"), index=False)

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(os.path.join(save_dir, "lstm_tuning_results.csv"), index=False)

    with open(os.path.join(save_dir, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return best_model, metrics, pred_df, best_history.history
