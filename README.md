# Analisis Sentimen #IndonesiaSadar

Proyek analisis sentimen opini publik pada media sosial terkait gerakan **#IndonesiaSadar** menggunakan pendekatan *Machine Learning* berbasis teks (**TF-IDF** + **Support Vector Machine / LinearSVC** dan **Multinomial Naive Bayes**).

---

## 📁 Struktur Direktori Proyek

Proyek ini telah distrukturkan secara rapi dan modular:

```text
sentiment-indonesiasadar/
├── data/
│   ├── raw/                                        # Dataset mentah asli (tidak diubah/ditimpa)
│   │   ├── #IndonesiaSadar👌 - Data Training.csv
│   │   └── #IndonesiaSadar👌 - Data Ujian.csv
│   └── processed/                                  # Hasil pembersihan & filtering data
│       └── data_training_clean.csv
│
├── models/                                         # Artefak model siap pakai (joblib)
│   ├── best_sentiment_model_svm.joblib
│   └── tfidf_vectorizer.joblib
│
├── notebooks/                                      # Eksplorasi & eksperimen interaktif (.ipynb)
│   └── 01_eksperimen_sentiment_analysis.ipynb
│
├── src/                                            # Skrip Python modular (.py) siap produksi
│   ├── __init__.py
│   ├── train.py                                    # Preprocessing, split, training & evaluasi
│   └── predict.py                                  # Skrip khusus untuk inferensi data baru / ujian
│
├── outputs/
│   ├── figures/                                    # Visualisasi grafik evaluasi
│   │   └── confusion_matrix_comparison.png
│   └── predictions/                                # File CSV hasil prediksi
│       ├── hasil_prediksi_20persen_data_uji.csv
│       └── hasil_prediksi_data_ujian.csv
│
├── .gitignore                                      # Abaikan folder venv/, __pycache__, dll.
├── requirements.txt                                # Daftar library Python yang dibutuhkan
└── README.md                                       # Panduan singkat cara menjalankan kode
```

---

## 🚀 Panduan Instalasi & Persiapan

### 1. Buat & Aktifkan Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

*(Atau jika menggunakan `uv`:)*
```bash
uv venv .venv
.\.venv\Scripts\activate
```

### 2. Pasang Dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Cara Menjalankan Kode

### A. Melatih Model & Evaluasi (`src/train.py`)

Skrip ini akan melakukan:
1. Membaca data mentah dari `data/raw/#IndonesiaSadar👌 - Data Training.csv`.
2. Membersihkan nilai kosong (NaN) dan memfilter label non-relevan.
3. Melakukan *Train-Test Split* 80:20 secara *stratified* (mencegah *data leakage*).
4. Melatih model **LinearSVC** dan **MultinomialNB** menggunakan ekstraksi fitur **TF-IDF**.
5. Menghasilkan laporan klasifikasi dan menyimpan visualisasi ke `outputs/figures/confusion_matrix_comparison.png`.
6. Menyimpan model terbaik ke `models/best_sentiment_model_svm.joblib` dan `models/tfidf_vectorizer.joblib`.
7. Mengekspor prediksi data uji ke `outputs/predictions/hasil_prediksi_20persen_data_uji.csv`.

**Perintah:**
```bash
python src/train.py
```

*Argumen opsional yang tersedia:*
- `--data`: Path ke file training (default: `data/raw/#IndonesiaSadar👌 - Data Training.csv`)
- `--model-out`: Lokasi simpan file model `.joblib`
- `--vec-out`: Lokasi simpan file vectorizer `.joblib`
- `--fig-out`: Lokasi simpan file grafik evaluasi
- `--pred-out`: Lokasi simpan file prediksi 20% data uji

---

### B. Inferensi pada Data Ujian Baru (`src/predict.py`)

Skrip khusus untuk melakukan inferensi / prediksi pada data baru yang belum berlabel (misalnya `#IndonesiaSadar👌 - Data Ujian.csv`).

**1. Prediksi batch dari file CSV:**
```bash
python src/predict.py
```
Hasil prediksi otomatis disimpan ke `outputs/predictions/hasil_prediksi_data_ujian.csv` dengan kolom baru `prediksi_label`.

*Argumen kustom:*
```bash
python src/predict.py --input "data/raw/#IndonesiaSadar👌 - Data Ujian.csv" --output "outputs/predictions/hasil_prediksi_data_ujian.csv"
```

**2. Prediksi kalimat tunggal secara langsung via CLI:**
```bash
python src/predict.py --text "Pemerintah tolong dengarkan aspirasi rakyat jelata"
```

---

### C. Eksperimen Interaktif di Jupyter Notebook

Untuk melihat analisis langkah demi langkah, visualisasi, dan eksplorasi data secara visual:

```bash
jupyter notebook notebooks/01_eksperimen_sentiment_analysis.ipynb
```
*(Atau buka langsung file notebook tersebut di VS Code / Antigravity IDE).*

---

## 📊 Hasil Evaluasi Model

| Model | Accuracy | Weighted F1-Score | Keterangan |
|---|---|---|---|
| **Support Vector Machine (LinearSVC)** | **~59.6%** | **~0.575** | **Model Terbaik (Dipilih)** |
| Naive Bayes (MultinomialNB) | ~55.9% | ~0.461 | - |

Model **LinearSVC** dipilih sebagai artefak produksi (`models/best_sentiment_model_svm.joblib`) karena memiliki performa klasifikasi F1-weighted yang lebih seimbang di seluruh kategori label (`Kontra`, `Pro`, `Ragu`, `Netral`).
