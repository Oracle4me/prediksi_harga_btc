import os
import json
import joblib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static", "image")
os.makedirs(IMAGE_DIR, exist_ok=True)

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

def save_prediction_plot(dates, actual, predicted, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))

    plt.plot(
        dates,
        actual,
        label="Actual"
    )

    plt.plot(
        dates,
        predicted,
        label="Predicted"
    )

    plt.title(filename.replace(".png", ""))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            BASE_DIR,
            "static",
            "image",
            filename
        ),
        bbox_inches="tight"
    )

    plt.close()
    
def save_prediction_and_metrics(model, metrics, pred_df, save_dir, model_filename, prediction_filename, metrics_filename):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, model_filename))
    pred_df.to_csv(os.path.join(save_dir, prediction_filename), index=False)

    with open(os.path.join(save_dir, metrics_filename), "w") as f:
        json.dump(metrics, f, indent=4)

def train_random_forest(X_train, X_test, y_train, y_test, dates_test, tune=False, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    if tune:
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [8, 12, 18, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        tscv = TimeSeriesSplit(n_splits=3)

        gs = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0
        )

        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_

    else:
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        best_params = {}

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, "Random Forest")
    metrics["best_params"] = best_params

    pred_df = pd.DataFrame({
        "Date": dates_test,
        "Actual": y_test,
        "Predicted": y_pred
    })

    save_prediction_plot(
        dates_test,
        y_test,
        y_pred,
        "rf_actual_vs_predicted.png"
    )

    save_prediction_and_metrics(
        model=model,
        metrics=metrics,
        pred_df=pred_df,
        save_dir=save_dir,
        model_filename="rf_model.pkl",
        prediction_filename="rf_predictions.csv",
        metrics_filename="rf_metrics.json"
    )

    return model, metrics, pred_df

def train_xgboost(X_train, X_test, y_train, y_test, dates_test, tune=False, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    if tune:
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0]
        }

        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )

        tscv = TimeSeriesSplit(n_splits=3)

        gs = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0
        )

        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_

    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        best_params = {}

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred, "XGBoost")
    metrics["best_params"] = best_params

    pred_df = pd.DataFrame({
        "Date": dates_test,
        "Actual": y_test,
        "Predicted": y_pred
    })

    save_prediction_plot(
        dates_test,
        y_test,
        y_pred,
        "xgb_actual_vs_predicted.png"
    )

    save_prediction_and_metrics(
        model=model,
        metrics=metrics,
        pred_df=pred_df,
        save_dir=save_dir,
        model_filename="xgb_model.pkl",
        prediction_filename="xgb_predictions.csv",
        metrics_filename="xgb_metrics.json"
    )

    return model, metrics, pred_df

def reshape_for_rnn(X):
    X = np.array(X)

    if len(X.shape) == 2:
        return X.reshape((X.shape[0], 1, X.shape[1]))

    return X

def train_simple_rnn(X_train, X_test, y_train, y_test, dates_test, save_dir="models", epochs=50, batch_size=16):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    os.makedirs(save_dir, exist_ok=True)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)
    y_test_array = np.array(y_test).flatten()

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_train_rnn = reshape_for_rnn(X_train_scaled)
    X_test_rnn = reshape_for_rnn(X_test_scaled)

    model = Sequential([
        Input(shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
        SimpleRNN(64, activation="tanh"),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.005),
        loss="huber"
    )

    callbacks = [
        EarlyStopping(monitor="loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-6)
    ]

    model.fit(
        X_train_rnn,
        y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False
    )

    y_pred_scaled = model.predict(X_test_rnn, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    metrics = compute_metrics(y_test_array, y_pred, "Simple RNN")
    metrics["best_params"] = {
        "units": 64,
        "dropout": 0.1,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.005
    }

    pred_df = pd.DataFrame({
        "Date": dates_test,
        "Actual": y_test_array,
        "Predicted": y_pred
    })

    save_prediction_plot(
        dates_test,
        y_test_array,
        y_pred,
        "rnn_actual_vs_predicted.png"
    )

    model.save(os.path.join(save_dir, "simple_rnn_model.keras"))
    joblib.dump(scaler_X, os.path.join(save_dir, "simple_rnn_scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(save_dir, "simple_rnn_scaler_y.pkl"))

    pred_df.to_csv(os.path.join(save_dir, "simple_rnn_predictions.csv"), index=False)

    with open(os.path.join(save_dir, "simple_rnn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return model, metrics, pred_df
