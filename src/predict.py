"""
Modul Inferensi Sentimen (predict.py)
=====================================
Skrip khusus untuk tim evaluasi (Imam) untuk melakukan inferensi / prediksi
label sentimen pada data baru atau data ujian yang belum berlabel.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
import joblib

# Root directory proyek
BASE_DIR = Path(__file__).resolve().parent.parent

# Default paths
DEFAULT_UNLABELED_FILE = BASE_DIR / "data" / "raw" / "#IndonesiaSadar👌 - Data Ujian.csv"
DEFAULT_MODEL_FILE = BASE_DIR / "models" / "best_sentiment_model_svm.joblib"
DEFAULT_VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
DEFAULT_OUTPUT_PREDICTION = BASE_DIR / "outputs" / "predictions" / "hasil_prediksi_data_ujian.csv"


def predict_unlabeled_data(
    input_filepath: Path | str = DEFAULT_UNLABELED_FILE,
    model_filepath: Path | str = DEFAULT_MODEL_FILE,
    vectorizer_filepath: Path | str = DEFAULT_VECTORIZER_FILE,
    output_filepath: Path | str = DEFAULT_OUTPUT_PREDICTION,
    text_column: str = 'komentar_clean'
) -> pd.DataFrame:
    """
    Memuat artefak model terbaik dan TfidfVectorizer (.joblib) untuk memprediksi
    file data baru yang belum memiliki label.
    """
    input_filepath = Path(input_filepath)
    model_filepath = Path(model_filepath)
    vectorizer_filepath = Path(vectorizer_filepath)
    output_filepath = Path(output_filepath)

    if not model_filepath.exists():
        raise FileNotFoundError(
            f"Artefak model tidak ditemukan di: {model_filepath}. "
            "Harap jalankan 'python src/train.py' terlebih dahulu."
        )
    if not vectorizer_filepath.exists():
        raise FileNotFoundError(
            f"Artefak vectorizer tidak ditemukan di: {vectorizer_filepath}. "
            "Harap jalankan 'python src/train.py' terlebih dahulu."
        )
    if not input_filepath.exists():
        raise FileNotFoundError(f"File data input tidak ditemukan di: {input_filepath}")

    print("=" * 60)
    print("INFERENSI ANALISIS SENTIMEN INDONESIA SADAR")
    print("=" * 60)
    print(f"[INFO] Memuat Model      : {model_filepath.name}")
    print(f"[INFO] Memuat Vectorizer : {vectorizer_filepath.name}")
    print(f"[INFO] File Input        : {input_filepath}")

    # 1. Muat model dan vectorizer
    loaded_model = joblib.load(model_filepath)
    loaded_vectorizer = joblib.load(vectorizer_filepath)

    # 2. Baca dataset
    if input_filepath.suffix.lower() in ['.xlsx', '.xls']:
        df_unlabeled = pd.read_excel(input_filepath)
    else:
        df_unlabeled = pd.read_csv(input_filepath)

    # Pastikan kolom teks tersedia
    if text_column not in df_unlabeled.columns:
        # Coba fallback ke 'komentar_asli' atau kolom teks pertama
        if 'komentar_asli' in df_unlabeled.columns:
            text_column = 'komentar_asli'
            print(f"[WARN] Kolom 'komentar_clean' tidak ada, beralih ke: '{text_column}'")
        else:
            raise KeyError(f"Kolom '{text_column}' tidak ditemukan dalam dataset: {list(df_unlabeled.columns)}")

    # 3. Tangani potensi nilai NaN dengan string kosong
    clean_texts = df_unlabeled[text_column].fillna('').astype(str).str.strip()

    # 4. Transformasi teks ke matriks TF-IDF dan lakukan prediksi
    X_new_tfidf = loaded_vectorizer.transform(clean_texts)
    df_unlabeled['prediksi_label'] = loaded_model.predict(X_new_tfidf)

    # 5. Simpan hasil prediksi ke file CSV
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    df_unlabeled.to_csv(output_filepath, index=False)

    print(f"\n[INFERENSI SELESAI] {len(df_unlabeled)} baris berhasil diprediksi!")
    print(f"[EXPORT] Hasil disimpan ke: {output_filepath}")
    print("\n[INFO] Distribusi Hasil Prediksi:")
    print(df_unlabeled['prediksi_label'].value_counts())

    return df_unlabeled


def predict_single_text(text: str, model_filepath: Path = DEFAULT_MODEL_FILE, vectorizer_filepath: Path = DEFAULT_VECTORIZER_FILE) -> str:
    """
    Fungsi utilitas untuk memprediksi satu kalimat teks secara langsung.
    """
    loaded_model = joblib.load(model_filepath)
    loaded_vectorizer = joblib.load(vectorizer_filepath)
    vec = loaded_vectorizer.transform([str(text)])
    pred = loaded_model.predict(vec)[0]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferensi Model Sentimen Indonesia Sadar untuk Imam")
    parser.add_argument("--input", type=str, default=str(DEFAULT_UNLABELED_FILE), help="Path ke data CSV/Excel yang akan diprediksi")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PREDICTION), help="Path output file hasil prediksi")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_FILE), help="Path file model .joblib")
    parser.add_argument("--vectorizer", type=str, default=str(DEFAULT_VECTORIZER_FILE), help="Path file vectorizer .joblib")
    parser.add_argument("--text", type=str, default=None, help="Teks kalimat tunggal yang ingin diprediksi langsung via CLI")

    args = parser.parse_args()

    if args.text:
        prediksi = predict_single_text(args.text, Path(args.model), Path(args.vectorizer))
        print(f"\nTeks    : {args.text}")
        print(f"Prediksi: {prediksi}")
    else:
        predict_unlabeled_data(
            input_filepath=Path(args.input),
            model_filepath=Path(args.model),
            vectorizer_filepath=Path(args.vectorizer),
            output_filepath=Path(args.output)
        )
