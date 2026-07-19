# Prediksi Harga Bitcoin Menggunakan LSTM, Simple RNN, Random Forest, dan XGBoost

Aplikasi ini digunakan untuk memprediksi harga Bitcoin menggunakan empat model machine learning dan deep learning, yaitu LSTM, Simple RNN, Random Forest, dan XGBoost. Model dievaluasi menggunakan MAE, RMSE, MAPE, dan R² untuk menentukan model dengan performa terbaik.

---

## 1. Fitur Aplikasi

- Mengambil data historis Bitcoin dari file CSV atau Yahoo Finance.
- Melatih tiga model prediksi:
  - LSTM
  - Simple RNN
  - XGBoost
  - Random Forest

- Menampilkan hasil evaluasi model:
  - MAE
  - RMSE
  - R² Score
  - MAPE

- Menampilkan grafik harga aktual dan hasil prediksi model.
- Menampilkan prediksi harga Bitcoin 1 hari ke depan beserta rentang estimasi harga.
- Dashboard berbasis Flask dengan tampilan web interaktif.

---

## 2. Struktur Folder

```text
prediksi_harga_btc/
│
├── data/
│   ├── btc_data.csv
│   ├── test_data.csv
│   └── train_data.csv
│
├── models/
│   ├── saved/
│   ├── lstm_model.py
│   └── ml_models.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── image/
│   │   └── .png
│   └── js/
│       └── app.js
│
├── templates/
│   └── index.html
│
├── utils/
│   └── preprocessing.py
│
├── app.py
├── train_all.py
├── requirements.txt
├── README.md
└── AlurPrediksiHarga.txt
└── Catatan Perhitungan & Penjelasan.txt
```

---

## 3. Persyaratan Sistem

Sebelum menjalankan project, pastikan sudah menginstall:

- Python 3.12
- pip
- Virtual environment Python
- Koneksi internet untuk mengambil data dari Yahoo Finance jika data belum tersedia

---

## 4. Instalasi Project

### 4.1. Clone atau buka folder project

Masuk ke folder project:

```bash
cd prediksi_harga_btc
```

### 4.2. Buat virtual environment

Windows:

```bash
python -m venv venv
```

Aktifkan virtual environment:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3. Install dependency

```bash
pip install -r requirements.txt
```

---

## 5. Menjalankan Training Model

Sebelum membuka dashboard, jalankan proses training terlebih dahulu:

```bash
python train_all.py
```

Proses ini akan melatih empat model:

1. LSTM
2. Simple RNN
3. Random Forest
4. XGBoost

Setelah training selesai, hasil model dan file evaluasi akan disimpan ke folder:

```text
models/hasil_train/
```

File yang dihasilkan antara lain:

```text
lstm_model.keras
rf_model.pkl
xgb_model.pkl
lstm_predictions.csv
rf_predictions.csv
xgb_predictions.csv
lstm_metrics.json
rf_metrics.json
xgb_metrics.json
training_summary.json
simple_rnn_model.keras
simple_rnn_predictions.csv
simple_rnn_metrics.json
```

---

## 6. Menjalankan Aplikasi Web

Setelah proses training selesai, jalankan aplikasi Flask:

```bash
python app.py
```

Kemudian buka browser dan akses:

```text
http://127.0.0.1:5000
```

Dashboard akan menampilkan:

- Harga Bitcoin terkini
- Metrik evaluasi model
- Grafik harga aktual dan harga prediksi
- Grafik prediksi LSTM, Simple RNN, Random Forest, dan XGBoost
- Prediksi harga Bitcoin 1 hari ke depan

---

## 7. Alur Penggunaan Singkat

Urutan penggunaan project:

```text
1. Install Python
2. Buat dan aktifkan virtual environment
3. Install requirements
4. Jalankan python train_all.py
5. Tunggu proses training selesai
6. Jalankan python app.py
7. Buka http://127.0.0.1:5000
```

---

## 8. Penjelasan Model

### 8.1. LSTM

LSTM digunakan sebagai model utama karena mampu mempelajari pola data deret waktu atau time-series. Model ini membaca sequence data historis Bitcoin untuk memprediksi harga hari berikutnya.

Pada penelitian ini, LSTM digunakan untuk mempelajari pola historis harga Bitcoin dan menghasilkan prediksi harga pada periode berikutnya.

### 8.2. Simple RNN

Simple RNN (Recurrent Neural Network) digunakan sebagai model pembanding berbasis deep learning untuk data deret waktu (time-series). Model ini memanfaatkan informasi dari periode sebelumnya untuk memprediksi harga Bitcoin pada periode berikutnya.

Dibandingkan LSTM, arsitektur Simple RNN lebih sederhana karena tidak memiliki mekanisme memory cell dan gate. Oleh karena itu, model ini lebih cepat dilatih, namun umumnya kurang mampu menangkap pola jangka panjang pada data historis yang kompleks.

Pada penelitian ini, Simple RNN digunakan untuk membandingkan performa model deep learning terhadap LSTM dalam memprediksi harga Bitcoin.

### 8.3. XGBoost

XGBoost digunakan sebagai model pembanding. Model ini sangat kuat untuk data tabular dengan fitur teknikal seperti harga Open, High, Low, Close, Volume, RSI, MACD, Moving Average, Momentum, dan Volatility.

XGBoost umumnya unggul dalam prediksi nilai harga secara numerik.

### 8.4. Random Forest

Random Forest digunakan sebagai model pembanding berbasis ensemble tree. Model ini bekerja dengan membangun banyak decision tree dan mengambil rata-rata hasil prediksi.

---

## 9. Metrik Evaluasi

### 9.1. MAE

MAE atau Mean Absolute Error digunakan untuk menghitung rata-rata selisih absolut antara harga aktual dan harga prediksi.

Semakin kecil nilai MAE, semakin baik performa model.

### 9.2. RMSE

RMSE atau Root Mean Squared Error digunakan untuk mengukur besar error prediksi dengan memberikan penalti lebih besar pada error yang besar.

Semakin kecil nilai RMSE, semakin baik performa model.

### 9.3. R² Score

R² digunakan untuk mengukur seberapa baik model mengikuti variasi harga aktual.

Semakin mendekati 1, maka model semakin baik dalam mengikuti pola harga aktual.

### 9.4. MAPE

MAPE atau Mean Absolute Percentage Error digunakan untuk mengetahui rata-rata persentase kesalahan prediksi terhadap harga aktual.

Semakin kecil nilai MAPE, semakin baik performa model.

---

## 10. Interpretasi Hasil

Hasil evaluasi model dianalisis menggunakan empat metrik, yaitu MAE, RMSE, MAPE, dan R² Score.

- MAE, RMSE, dan MAPE digunakan untuk mengukur tingkat kesalahan prediksi. Semakin kecil nilainya, semakin baik performa model.
- R² Score digunakan untuk mengukur kemampuan model dalam mengikuti pola data aktual. Semakin mendekati 1, semakin baik kemampuan model dalam menjelaskan variasi harga Bitcoin.

Pada penelitian ini, model terbaik ditentukan berdasarkan nilai R² Score tertinggi. Model dengan nilai R² tertinggi dianggap memiliki kemampuan terbaik dalam mempelajari pola historis harga Bitcoin dan menghasilkan prediksi yang paling akurat.

Sebagai contoh, jika model LSTM memperoleh nilai R² yang lebih tinggi dibandingkan Simple RNN, Random Forest, dan XGBoost, maka LSTM dipilih sebagai model utama untuk menghasilkan prediksi harga Bitcoin pada dashboard aplikasi.

## 11. Catatan Penting

- Model memprediksi harga Bitcoin dalam satuan USD.
- Dashboard mengonversi harga ke IDR menggunakan kurs USD/IDR terbaru.
- Prediksi 1 hari ke depan menggunakan model LSTM.
- Hasil prediksi bukan merupakan saran investasi.
- Cryptocurrency memiliki volatilitas tinggi sehingga hasil prediksi dapat berubah mengikuti kondisi pasar.

---

## 12. Troubleshooting

### 12.1. Module tidak ditemukan

Jika muncul error seperti:

```text
ModuleNotFoundError
```

Pastikan dependency sudah diinstall:

```bash
pip install -r requirements.txt
```

### 12.2. Model belum tersedia

Jika dashboard belum menampilkan hasil prediksi, jalankan training terlebih dahulu:

```bash
python train_all.py
```

### 12.3. Aplikasi tidak bisa dibuka

Pastikan Flask sudah berjalan:

```bash
python app.py
```

Lalu buka:

```text
http://127.0.0.1:5000
```

### 12.4. Data atau model tidak sesuai

Jika terjadi error karena jumlah fitur model tidak cocok, hapus isi folder berikut:

```text
models/hasil_train/
```

Kemudian jalankan ulang:

```bash
python train_all.py
```

---

## 13. Perintah Utama

Training model:

```bash
python train_all.py
```

Menjalankan dashboard:

```bash
python app.py
```

Membuka aplikasi:

```text
http://127.0.0.1:5000
```

---

## 14. Disclaimer

Aplikasi ini dibuat untuk kebutuhan penelitian dan pembelajaran. Hasil prediksi yang ditampilkan oleh model tidak menjamin pergerakan harga Bitcoin di masa depan dan bukan merupakan saran investasi.
