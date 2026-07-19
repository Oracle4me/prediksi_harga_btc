import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "btc_usd_2024_2026.csv")

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = None
BTC_SYMBOL = "BTC-USD"


def load_or_fetch_data(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE, force_update=False):
    """
    Memuat data historis Bitcoin (BTC-USD).

    Sumber data:
    - CSV lokal (prioritas)
    - Yahoo Finance (jika file tidak tersedia)

    Output:
    - Data OHLCV (Open, High, Low, Close, Volume)
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE) and not force_update:
        print("\n" + "="*80)
        print("Tahap 1: Memuat Data BTC")
        print("="*80)
        print(f"[OK] Membaca data dari file lokal: {DATA_FILE}")
        
        df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        
        print(f"[OK] Total baris data (sebelum filter): {len(df)} baris")
        print(f"[OK] Kolom data: {list(df.columns)}")

        if start is not None:
            df = df[df.index >= pd.to_datetime(start)]
            print(f"[OK] Filter data dari tanggal: {start}")

        if end is not None:
            df = df[df.index <= pd.to_datetime(end)]
            print(f"[OK] Filter data hingga tanggal: {end}")
        
        print(f"[OK] Total baris setelah filter: {len(df)} baris")
        print(f"[OK] Range tanggal: {df.index.min().date()} hingga {df.index.max().date()}")
        print(f"[OK] Nilai Close: Min={df['Close'].min():.2f} | Max={df['Close'].max():.2f} | Mean={df['Close'].mean():.2f}")
        
        return df

    try:
        import yfinance as yf

        print("\n" + "="*80)
        print("TAHAP 1: MEMUAT DATA BITCOIN (dari Yahoo Finance)")
        print("="*80)
        print("[WARNING] File lokal tidak ditemukan, download dari Yahoo Finance...")
        print(f"[OK] Symbol: {BTC_SYMBOL}")
        print(f"[OK] Range: {start} hingga {end}")
        
        df = yf.download(
            BTC_SYMBOL,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        df = df.dropna()
        df.index.name = "Date"
        
        print(f"[OK] Total baris data: {len(df)} baris")
        print(f"[OK] Range tanggal: {df.index.min().date()} hingga {df.index.max().date()}")

        df.to_csv(DATA_FILE)
        print(f"[OK] Data disimpan ke: {DATA_FILE}")
        
        return df

    except Exception as e:
        print(f"[ERROR] {e}")
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
            return df.set_index("Date").sort_index()

        return None


def add_technical_indicators(df):
    """
    Menambahkan fitur teknikal untuk membantu model mengenali
    pola harga, momentum, volatilitas, volume, dan trend pasar.

    Kelompok fitur:
    - Return
    - Momentum
    - Moving Average (MA)
    - Volatility
    - RSI
    - MACD
    - ATR
    - Volume Indicator
    """

    print("\n" + "="*80)
    print("TAHAP 2: FEATURE ENGINEERING (Menambah Technical Indicators)")
    print("="*80)
    print(f"Data input: {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Kolom awal: {list(df.columns)}\n")

    df = df.copy()

    print("Menambah fitur-fitur teknikal:\n")
    
    # Return historis untuk menangkap perubahan harga jangka pendek.
    print("[1] RETURN (Perubahan Persentase Harga)")
    print("    Fitur: Return_1, Return_2, Return_3, Return_7, Return_14")
    print("    Tujuan: Menangkap momentum perubahan harga jangka pendek\n")
    
    df["Return_1"] = df["Close"].pct_change()
    df["Return_2"] = df["Close"].pct_change(2)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Return_7"] = df["Close"].pct_change(7)
    df["Return_14"] = df["Close"].pct_change(14)

    # Mengukur kekuatan pergerakan harga dibanding beberapa hari sebelumnya.
    print("[2] MOMENTUM (Perubahan Nilai Absolut Harga)")
    print("    Fitur: Momentum_3, Momentum_7, Momentum_14")
    print("    Tujuan: Mengukur kekuatan pergerakan harga\n")
    
    df["Momentum_3"] = df["Close"] - df["Close"].shift(3)
    df["Momentum_7"] = df["Close"] - df["Close"].shift(7)
    df["Momentum_14"] = df["Close"] - df["Close"].shift(14)

    # Trend harga rata-rata jangka pendek hingga menengah.
    print("[3] MOVING AVERAGE (MA) - Rata-rata Harga Bergerak")
    print("    Fitur: MA_7, MA_14, MA_30, MA_60")
    print("    Tujuan: Mendeteksi trend harga jangka pendek hingga menengah\n")
    
    df["MA_7"] = df["Close"].rolling(7).mean()
    df["MA_14"] = df["Close"].rolling(14).mean()
    df["MA_30"] = df["Close"].rolling(30).mean()
    df["MA_60"] = df["Close"].rolling(60).mean()

    # Kemiringan Moving Average untuk mengukur arah trend.
    print("[4] MA SLOPE - Kemiringan Moving Average")
    print("    Fitur: MA_7_Slope, MA_14_Slope, MA_30_Slope, MA_60_Slope")
    print("    Tujuan: Mengukur perubahan trend (naik/turun)\n")
    
    df["MA_7_Slope"] = df["MA_7"].diff()
    df["MA_14_Slope"] = df["MA_14"].diff()
    df["MA_30_Slope"] = df["MA_30"].diff()
    df["MA_60_Slope"] = df["MA_60"].diff()

    # Posisi harga relatif terhadap Moving Average.
    print("[5] CLOSE-MA RATIO - Posisi Harga vs MA")
    print("    Fitur: Close_MA7_Ratio, Close_MA14_Ratio, Close_MA30_Ratio")
    print("    Tujuan: Mengetahui harga ada di atas/bawah rata-rata\n")
    
    df["Close_MA7_Ratio"] = df["Close"] / df["MA_7"]
    df["Close_MA14_Ratio"] = df["Close"] / df["MA_14"]
    df["Close_MA30_Ratio"] = df["Close"] / df["MA_30"]

    # Tingkat fluktuasi harga dalam periode tertentu.
    print("[6] VOLATILITY - Tingkat Fluktuasi Harga")
    print("    Fitur: Volatility_7, Volatility_14, Volatility_30")
    print("    Tujuan: Mengukur risiko/ketidakstabilan harga\n")
    
    df["Volatility_7"] = df["Return_1"].rolling(7).std()
    df["Volatility_14"] = df["Return_1"].rolling(14).std()
    df["Volatility_30"] = df["Return_1"].rolling(30).std()

    # RSI - Relative Strength Index
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)

    print("[7] RSI (Relative Strength Index)")
    print("    Fitur: RSI_14")
    print("    Tujuan: Deteksi kondisi overbought/oversold (0-100)\n")
    
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    print("[8] MACD (Moving Average Convergence Divergence)")
    print("    Fitur: MACD, MACD_Signal, MACD_Hist")
    print("    Tujuan: Mengukur momentum dan perubahan trend\n")
    
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    print("[9] ATR (Average True Range)")
    print("    Fitur: ATR_14")
    print("    Tujuan: Mengukur volatilitas pasar\n")
    
    df["ATR_14"] = true_range.rolling(14).mean()

    # Volume Indicators
    print("[10] VOLUME INDICATORS - Indikator Volume Transaksi")
    print("     Fitur: Volume_Change, Volume_MA_7, Volume_MA_14, Volume_Ratio_7")
    print("     Tujuan: Deteksi kekuatan trend melalui volume transaksi\n")
    
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_7"] = df["Volume"].rolling(7).mean()
    df["Volume_MA_14"] = df["Volume"].rolling(14).mean()
    df["Volume_Ratio_7"] = df["Volume"] / df["Volume_MA_7"]

    print("="*80)
    print("Data Cleaning (Pembersihan Data)")
    print("="*80)
    
    nan_before = df.isna().sum().sum()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    nan_after = df.isna().sum().sum()
    
    print("[OK] Menghapus nilai NaN (Not a Number)")
    print("[OK] Mengganti Inf dan -Inf dengan NaN lalu dihapus")
    print(f"[OK] Total baris setelah cleaning: {len(df)} baris")
    print(f"[OK] Total kolom: {df.shape[1]} kolom")

    return df


def get_feature_columns():
    return [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1", "Return_2", "Return_3", "Return_7", "Return_14",
        "Momentum_3", "Momentum_7", "Momentum_14",
        "MA_7", "MA_14", "MA_30", "MA_60",
        "MA_7_Slope", "MA_14_Slope", "MA_30_Slope", "MA_60_Slope",
        "Close_MA7_Ratio", "Close_MA14_Ratio", "Close_MA30_Ratio",
        "Volatility_7", "Volatility_14", "Volatility_30",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "ATR_14",
        "Volume_Change", "Volume_MA_7", "Volume_MA_14", "Volume_Ratio_7"
    ]


def save_train_test_data(train_df, test_df):
    train_df.to_csv(
        os.path.join(DATA_DIR, "train_data.csv"),
        index=True
    )

    test_df.to_csv(
        os.path.join(DATA_DIR, "test_data.csv"),
        index=True
    )

def prepare_compare_data(df, test_size=0.2, forecast_horizon=1):
    """
    Menyiapkan data untuk Random Forest, XGBoost, dan Simple RNN.

    Target:
    - Harga Close pada horizon berikutnya (regresi numerik)

    Label:
    - Numerik (bukan kategorikal)

    Pembagian data:
    - Train = 80%
    - Test = 20%

    Tidak dilakukan normalisasi karena model tree-based
    (Random Forest dan XGBoost) tidak memerlukannya.
    """

    df_ind = add_technical_indicators(df)
    feature_cols = get_feature_columns()

    print("\n" + "="*80)
    print("TAHAP 3.1: MENDEFINISIKAN TARGET (y)")
    print("="*80)
    print("Tujuan: Memprediksi harga Close hari berikutnya (regresi numerik)")
    print(f"[OK] Forecast Horizon: {forecast_horizon} hari ke depan")
    print(f"[OK] Target = Close[t+{forecast_horizon}] (shift -{forecast_horizon})")

    # Target = harga Close hari berikutnya.
    df_ind["Target"] = df_ind["Close"].shift(-forecast_horizon)
    df_ind = df_ind.dropna()

    X = df_ind[feature_cols].values
    y = df_ind["Target"].values
    dates = df_ind.index

    print(f"[OK] Total sample setelah shift dan dropna: {len(df_ind)} baris")
    print(f"[OK] Target value: Min={y.min():.2f} | Max={y.max():.2f} | Mean={y.mean():.2f}")

    # Time-series split tanpa shuffle untuk menghindari data leakage.
    print("\n" + "="*80)
    print("TAHAP 3.2: PEMBAGIAN DATA (Train-Test Split)")
    print("="*80)
    print("PENTING: Time-Series Split (tanpa shuffle untuk hindari data leakage)")
    print(f"[OK] Train Size: {1-test_size:.0%} ({int(len(df_ind)*(1-test_size))} baris)")
    print(f"[OK] Test Size:  {test_size:.0%} ({int(len(df_ind)*test_size)} baris)")

    split_idx = int(len(df_ind) * (1 - test_size))

    train_df = df_ind.iloc[:split_idx].copy()
    test_df = df_ind.iloc[split_idx:].copy()

    # Simpan pembagian data train dan test
    save_train_test_data(train_df, test_df)

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    dates_test = dates[split_idx:]

    print(f"[OK] Train data: {X_train.shape[0]} sampel x {X_train.shape[1]} fitur")
    print(f"[OK] Test data:  {X_test.shape[0]} sampel x {X_test.shape[1]} fitur")
    print(f"[OK] Training: {train_df.index.min().date()} hingga {train_df.index.max().date()}")
    print(f"[OK] Testing:  {test_df.index.min().date()} hingga {test_df.index.max().date()}")

    print("\n" + "="*80)
    print("NORMALISASI: Tidak Dilakukan Pada RF & XGBoost")
    print("="*80)
    print("Alasan:")
    print("  - Random Forest & XGBoost adalah tree-based models")
    print("  - Tree-based models TIDAK sensitif terhadap skala fitur")
    print("  - Normalisasi akan membuang informasi tetapi tidak meningkatkan akurasi")
    print("  - Justru normalisasi bisa memperlambat training")

    print("\n" + "="*80)
    print(f"Features Columns (Total: {len(feature_cols)})")
    print("="*80)
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    print("\n" + "="*80)
    print("OUTPUT prepare_compare_data()")
    print("="*80)
    print(f"[OK] X_train: shape {X_train.shape}")
    print(f"[OK] X_test:  shape {X_test.shape}")
    print(f"[OK] y_train: shape {y_train.shape}")
    print(f"[OK] y_test:  shape {y_test.shape}")
    print(f"[OK] dates_test: shape {dates_test.shape}")
    print(f"[OK] feature_cols: {len(feature_cols)} fitur")

    return X_train, X_test, y_train, y_test, dates_test, feature_cols


def prepare_lstm_data(df, seq_length=45, test_size=0.2, forecast_horizon=1):
    """
    Menyiapkan data sequence untuk model LSTM.

    Target:
    - Harga Close pada horizon berikutnya (regresi numerik)

    Label:
    - Numerik (continuous value)

    Sequence:
    - Default 45 hari observasi sebelumnya

    Pembagian data:
    - Train = 80%
    - Test = 20%

    Normalisasi:
    - MinMaxScaler untuk fitur
    - MinMaxScaler untuk target
    """

    print("\n" + "="*80)
    print("TAHAP 4: PREPARE DATA UNTUK LSTM (Deep Learning)")
    print("="*80)

    df_ind = add_technical_indicators(df)
    feature_cols = get_feature_columns()

    print("\n" + "="*80)
    print("TAHAP 4.1: MENDEFINISIKAN TARGET (y)")
    print("="*80)
    print("Tujuan: Memprediksi harga Close hari berikutnya (regresi numerik)")
    print(f"[OK] Forecast Horizon: {forecast_horizon} hari ke depan")

    df_ind["Target"] = df_ind["Close"].shift(-forecast_horizon)
    df_ind = df_ind.dropna()

    feature_values = df_ind[feature_cols].values
    target_values = df_ind[["Target"]].values
    dates = df_ind.index

    print(f"[OK] Total data setelah shift dan dropna: {len(df_ind)} baris")

    split_idx = int(len(df_ind) * (1 - test_size))
    
    print("\n" + "="*80)
    print("TAHAP 4.2: PEMBAGIAN DATA (Train-Test Split)")
    print("="*80)
    print("PENTING: Time-Series Split (tanpa shuffle untuk hindari data leakage)")
    print(f"[OK] Train Size: {1-test_size:.0%} ({split_idx} baris)")
    print(f"[OK] Test Size:  {test_size:.0%} ({len(df_ind)-split_idx} baris)")
    print(f"[OK] Training: {df_ind.index[:split_idx].min().date()} hingga {df_ind.index[:split_idx].max().date()}")
    print(f"[OK] Testing:  {df_ind.index[split_idx:].min().date()} hingga {df_ind.index[split_idx:].max().date()}")
    
    # Normalisasi fitur ke rentang 0–1.
    print("\n" + "="*80)
    print("TAHAP 4.3: NORMALISASI FITUR")
    print("="*80)
    print("Mengapa normalisasi diperlukan untuk LSTM?")
    print("  - LSTM adalah neural network yang sensitif terhadap skala")
    print("  - Normalisasi membantu gradient descent konvergen lebih cepat")
    print("  - Range 0-1 optimal untuk aktivasi sigmoid/relu")
    print("  - Menghindari masalah vanishing/exploding gradients")
    
    scaler = MinMaxScaler()
    print(f"\n[OK] MinMaxScaler(feature_range=(0, 1))")
    print(f"[OK] Fitur awal (feature_values):")
    print(f"     Min per fitur: {feature_values.min(axis=0)[:5]} ... (ditampilkan 5 pertama)")
    print(f"     Max per fitur: {feature_values.max(axis=0)[:5]} ... (ditampilkan 5 pertama)")
    
    # Normalisasi target harga ke rentang 0–1.
    print(f"\n[OK] MinMaxScaler(feature_range=(0, 1)) untuk Target")
    print(f"[OK] Target awal (Close prices):")
    print(f"     Min: {target_values.min():.2f}")
    print(f"     Max: {target_values.max():.2f}")
    
    close_scaler = MinMaxScaler()

    train_features = feature_values[:split_idx]
    train_targets = target_values[:split_idx]

    scaler.fit(train_features)
    close_scaler.fit(train_targets)

    scaled_features = scaler.transform(feature_values)
    scaled_targets = close_scaler.transform(target_values).flatten()

    print(f"\n[OK] Fitur setelah normalisasi:")
    print(f"     Min: {scaled_features.min(axis=0)[:5]} ... (semua ~0)")
    print(f"     Max: {scaled_features.max(axis=0)[:5]} ... (semua ~1)")
    print(f"\n[OK] Target setelah normalisasi:")
    print(f"     Min: {scaled_targets.min():.4f}")
    print(f"     Max: {scaled_targets.max():.4f}")

    # Sequence Creation
    print("\n" + "="*80)
    print("TAHAP 4.4: MEMBUAT SEQUENCE (Sliding Window)")
    print("="*80)
    print("PENTING: LSTM adalah sequence model yang memerlukan urutan temporal")
    print(f"[OK] Sequence Length: {seq_length} hari observasi sebelumnya")
    print(f"[OK] Prediksi: 1 hari ke depan")
    print(f"[OK] Setiap sample berisi {seq_length} timesteps x {len(feature_cols)} fitur")
    
    X, y, seq_dates = [], [], []

    for i in range(seq_length, len(df_ind)):
        X.append(scaled_features[i - seq_length:i])
        y.append(scaled_targets[i])
        seq_dates.append(dates[i])

    X = np.array(X)
    y = np.array(y)
    seq_dates = pd.DatetimeIndex(seq_dates)

    print(f"[OK] Total sequence yang dibuat: {len(X)}")
    print(f"[OK] Ukuran X: {X.shape} (samples, timesteps, features)")
    print(f"[OK] Ukuran y: {y.shape} (targets)")

    train_cutoff_date = dates[split_idx]

    print("\n" + "="*80)
    print("TAHAP 4.5: SPLIT SEQUENCE MENJADI TRAIN-TEST")
    print("="*80)

    train_mask = seq_dates < train_cutoff_date
    test_mask = seq_dates >= train_cutoff_date

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    dates_test = seq_dates[test_mask]

    print(f"[OK] Train cutoff date: {train_cutoff_date.date()}")
    print(f"[OK] X_train: shape {X_train.shape}")
    print(f"[OK] X_test:  shape {X_test.shape}")
    print(f"[OK] y_train: shape {y_train.shape}")
    print(f"[OK] y_test:  shape {y_test.shape}")
    print(f"[OK] dates_test: {len(dates_test)} sampel")

    print("\n" + "="*80)
    print("RINGKASAN DATA LSTM")
    print("="*80)
    print(f"[OK] Jumlah feature: {len(feature_cols)}")
    print(f"[OK] Sequence length: {seq_length}")
    print(f"[OK] Forecast horizon: {forecast_horizon}")
    print(f"[OK] Total samples: {len(X)}")
    print(f"[OK] Training samples: {len(X_train)} x ({seq_length}, {len(feature_cols)})")
    print(f"[OK] Testing samples: {len(X_test)} x ({seq_length}, {len(feature_cols)})")
    print(f"[OK] Scalers: MinMaxScaler(feature_range=(0,1))")
    print(f"     - scaler: untuk normalisasi fitur")
    print(f"     - close_scaler: untuk normalisasi/denormalisasi target harga")

    return X_train, X_test, y_train, y_test, scaler, close_scaler, dates_test, feature_cols


# main
if __name__ == "__main__":
    print("\n")
    print("="*80)
    print("ALUR PREPROCESSING DATA BTC")
    print("="*80)

    # TAHAP 1: Load Data
    print("\n\nMulai Preprocessing...\n")
    
    df = load_or_fetch_data(
        start="2024-01-01",
        end=None,
        force_update=False
    )

    if df is None:
        print("[ERROR] Tidak bisa memuat data!")
        exit(1)

    X_train_rf, X_test_rf, y_train_rf, y_test_rf, dates_test_rf, feature_cols = prepare_compare_data(
        df, 
        test_size=0.2,
        forecast_horizon=1
    )

    print("\n[OK] Data siap untuk Random Forest & XGBoost")
    print(f"     X_train shape: {X_train_rf.shape}")
    print(f"     X_test shape: {X_test_rf.shape}")
    print(f"     y_train shape: {y_train_rf.shape}")
    print(f"     y_test shape: {y_test_rf.shape}")

    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler, close_scaler, dates_test_lstm, _ = prepare_lstm_data(
        df,
        seq_length=45,
        test_size=0.2,
        forecast_horizon=1
    )

    print("\n[OK] Data siap untuk LSTM")
    print(f"     X_train shape: {X_train_lstm.shape}")
    print(f"     X_test shape: {X_test_lstm.shape}")
    print(f"     y_train shape: {y_train_lstm.shape}")
    print(f"     y_test shape: {y_test_lstm.shape}")

    print("\n\n" + "="*80)
    print("Hasil Ringkasan Preprocessing")
    print("="*80)
    
    print("\nPerbandingan Kedua Alur:\n")
    
    print("=" * 70)
    print("TREE-BASED MODELS (Random Forest, XGBoost)")
    print("=" * 70)
    print(f"Feature format: 2D (samples x features)")
    print(f"  X_train: {X_train_rf.shape}")
    print(f"  X_test:  {X_test_rf.shape}")
    print(f"\nNormalisasi: TIDAK (tree-based tidak sensitif skala)")
    print(f"\nKeuntungan:")
    print(f"  - Cepat training")
    print(f"  - Interpretable (feature importance)")
    print(f"  - Cocok untuk feature importance analysis")

    print("\n" + "=" * 70)
    print("LSTM (Deep Learning)")
    print("=" * 70)
    print(f"Sequence format: 3D (samples x timesteps x features)")
    print(f"  X_train: {X_train_lstm.shape}")
    print(f"  X_test:  {X_test_lstm.shape}")
    print(f"\nNormalisasi: YA (MinMaxScaler 0-1)")
    print(f"  - scaler: untuk fitur")
    print(f"  - close_scaler: untuk target/denormalisasi output")
    print(f"\nKeuntungan:")
    print(f"  - Capture temporal dependencies")
    print(f"  - Handle sequential data")
    print(f"  - Potensi akurasi lebih tinggi")

    print("\n\n" + "="*80)
    print("Checklist Data ")
    print("="*80)
    
    checklist = [
        "Data loading dari file CSV",
        "Data cleaning (remove NaN, Inf)",
        "Feature engineering (technical indicators)",
        "Train-test split (80-20, time-series aware)",
        "Normalisasi untuk LSTM (MinMaxScaler)",
        "Sequence creation untuk LSTM (45 timesteps)",
        "Separate scalers untuk fitur dan target",
    ]
    
    for item in checklist:
        print(f"[OK] {item}")

    print("\n\n" + "="*80)
    print("Siap untuk memasuki training model!")
    print("="*80)
    print("\nGunakan output data untuk:")
    print("  1. Random Forest & XGBoost: gunakan X_train_rf, X_test_rf, dll")
    print("  2. LSTM: gunakan X_train_lstm, X_test_lstm, scaler, close_scaler")

    print("\nJalankan `python train_all.py` untuk memulai training.\n")
    print("\n")