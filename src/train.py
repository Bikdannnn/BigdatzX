"""
Modul Pelatihan & Evaluasi Model Analisis Sentimen Indonesia Sadar
==================================================================
Melakukan data loading, pembersihan, split data 80:20 (stratified),
ekstraksi fitur TF-IDF, pelatihan LinearSVC & MultinomialNB,
evaluasi & visualisasi confusion matrix, serta penyimpanan artefak model.
"""

import os
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Root directory proyek
BASE_DIR = Path(__file__).resolve().parent.parent

# Default file paths
DEFAULT_TRAIN_FILE = BASE_DIR / "data" / "raw" / "#IndonesiaSadar👌 - Data Training.csv"
DEFAULT_PROCESSED_FILE = BASE_DIR / "data" / "processed" / "data_training_clean.csv"
DEFAULT_MODEL_FILE = BASE_DIR / "models" / "best_sentiment_model_svm.joblib"
DEFAULT_VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
DEFAULT_FIG_FILE = BASE_DIR / "outputs" / "figures" / "confusion_matrix_comparison.png"
DEFAULT_PRED_20_FILE = BASE_DIR / "outputs" / "predictions" / "hasil_prediksi_20persen_data_uji.csv"


def load_and_preprocess_labeled_data(filepath: Path | str) -> pd.DataFrame:
    """
    Memuat data berlabel (CSV/Excel), membersihkan nilai kosong (NaN) pada
    kolom komentar_clean, dan memfilter keluar label 'Tidak Relevan'.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File data latih tidak ditemukan di: {filepath}")

    # Deteksi format file secara otomatis
    if filepath.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
        
    print(f"[INFO] Jumlah data awal: {len(df)} baris")
    
    # 1. Hapus baris dengan nilai NaN pada kolom 'komentar_clean' atau 'label'
    df = df.dropna(subset=['komentar_clean', 'label']).copy()
    
    # 2. Pastikan tipe string dan buang baris yang hanya berisi spasi kosong
    df['komentar_clean'] = df['komentar_clean'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['komentar_clean'] != '']
    
    # 3. Filter dan buang seluruh variasi label 'tidak relevan' (case-insensitive)
    irrelevant_labels = ['tidak relevan', 'tidak_relevan', 'irrelevant', 'tdk relevan']
    mask_irrelevant = df['label'].str.lower().isin(irrelevant_labels)
    df_clean = df[~mask_irrelevant].copy().reset_index(drop=True)
    
    print(f"[INFO] Jumlah data setelah hapus 'Tidak Relevan' & NaN: {len(df_clean)} baris")
    print("\n[INFO] Distribusi Label Bersih:")
    print(df_clean['label'].value_counts())
    
    return df_clean


def train_and_evaluate(
    train_path: Path = DEFAULT_TRAIN_FILE,
    model_output_path: Path = DEFAULT_MODEL_FILE,
    vectorizer_output_path: Path = DEFAULT_VECTORIZER_FILE,
    fig_output_path: Path = DEFAULT_FIG_FILE,
    pred_20_output_path: Path = DEFAULT_PRED_20_FILE,
    save_processed: bool = True
):
    """
    Pipeline lengkap training, evaluasi, visualisasi, dan export artefak.
    """
    # 1. Preprocessing Data
    print("=" * 60)
    print("1. MEMUAT & PREPROCESSING DATASET")
    print("=" * 60)
    df_clean = load_and_preprocess_labeled_data(train_path)

    # Simpan dataset bersih ke data/processed jika diminta
    if save_processed:
        DEFAULT_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(DEFAULT_PROCESSED_FILE, index=False)
        print(f"[INFO] Data bersih disimpan ke: {DEFAULT_PROCESSED_FILE}")

    # 2. Train-Test Split (80:20) & TF-IDF Vectorization
    print("\n" + "=" * 60)
    print("2. PEMBAGIAN DATA (80:20 STRATIFIED) & TF-IDF VECTORIZATION")
    print("=" * 60)
    X = df_clean['komentar_clean']
    y = df_clean['label']
    comment_ids = df_clean['comment_id']

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, comment_ids,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"[INFO] Ukuran Data Latih (80%): {len(X_train)} baris")
    print(f"[INFO] Ukuran Data Uji (20%)  : {len(X_test)} baris")

    # Fit TfidfVectorizer HANYA pada X_train untuk mencegah kebocoran data (data leakage)
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print(f"[INFO] Ukuran Kosakata TF-IDF : {len(tfidf_vectorizer.vocabulary_)} token unik")

    # 3. Pelatihan Model
    print("\n" + "=" * 60)
    print("3. PELATIHAN MODEL (LinearSVC & MultinomialNB)")
    print("=" * 60)
    # a. LinearSVC
    svm_model = LinearSVC(random_state=42, dual='auto')
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    # b. MultinomialNB
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    # 4. Evaluasi & Perbandingan
    print("\n" + "=" * 60)
    print("4. EVALUASI HASIL PREDIKSI")
    print("=" * 60)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

    acc_nb = accuracy_score(y_test, y_pred_nb)
    f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

    print("\n--- LAPORAN KLASIFIKASI: SUPPORT VECTOR MACHINE (LinearSVC) ---")
    print(f"Accuracy: {acc_svm:.4f} | Weighted F1-Score: {f1_svm:.4f}\n")
    print(classification_report(y_test, y_pred_svm, zero_division=0))

    print("\n--- LAPORAN KLASIFIKASI: NAIVE BAYES (MultinomialNB) ---")
    print(f"Accuracy: {acc_nb:.4f} | Weighted F1-Score: {f1_nb:.4f}\n")
    print(classification_report(y_test, y_pred_nb, zero_division=0))

    # Visualisasi Confusion Matrix
    labels_order = sorted(y.unique())
    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=labels_order)
    cm_nb = confusion_matrix(y_test, y_pred_nb, labels=labels_order)

    fig_output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='#0A1722')
    for ax, cm, title in zip(
        axes,
        [cm_svm, cm_nb],
        [f'SVM (LinearSVC)\nAccuracy: {acc_svm:.2%} | F1-Weighted: {f1_svm:.2%}',
         f'Naive Bayes (MultinomialNB)\nAccuracy: {acc_nb:.2%} | F1-Weighted: {f1_nb:.2%}']
    ):
        ax.set_facecolor('#0A1722')
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
            xticklabels=labels_order, yticklabels=labels_order,
            ax=ax, annot_kws={'size': 12, 'weight': 'bold'}
        )
        ax.set_title(title, color='#09D9F3', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Prediksi Model', color='white', fontsize=11)
        ax.set_ylabel('Label Asli (Aktual)', color='white', fontsize=11)
        ax.tick_params(colors='white', labelsize=10)

    plt.suptitle('Perbandingan Confusion Matrix pada 20% Data Uji', color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fig_output_path, dpi=300, bbox_inches='tight', facecolor='#0A1722')
    plt.close()
    print(f"[VISUALISASI] Confusion matrix disimpan ke: {fig_output_path}")

    # Pilih model terbaik berdasarkan F1-Score
    if f1_svm >= f1_nb:
        best_model = svm_model
        best_model_name = "SVM (LinearSVC)"
    else:
        best_model = nb_model
        best_model_name = "Naive Bayes (MultinomialNB)"

    print(f"\n[KESIMPULAN] Model Terbaik: {best_model_name} (F1-Weighted: {max(f1_svm, f1_nb):.4f})")

    # 5. Export Hasil & Model
    print("\n" + "=" * 60)
    print("5. MENYIMPAN HASIL PREDIKSI 20% DATA UJI & ARTEFAK MODEL")
    print("=" * 60)
    pred_20_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_evaluasi_20 = pd.DataFrame({
        'comment_id': id_test.values,
        'komentar_clean': X_test.values,
        'label_asli': y_test.values,
        'prediksi_svm': y_pred_svm,
        'prediksi_naive_bayes': y_pred_nb
    })
    df_evaluasi_20.to_csv(pred_20_output_path, index=False)
    print(f"[EXPORT] Prediksi 20% data uji : {pred_20_output_path}")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_output_path)
    joblib.dump(tfidf_vectorizer, vectorizer_output_path)
    print(f"[EXPORT] Model terbaik         : {model_output_path}")
    print(f"[EXPORT] TF-IDF Vectorizer     : {vectorizer_output_path}")
    print("\n[SUKSES] Training & evaluasi selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pelatihan Model Analisis Sentimen Indonesia Sadar")
    parser.add_argument("--data", type=str, default=str(DEFAULT_TRAIN_FILE), help="Path ke data training CSV/Excel")
    parser.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL_FILE), help="Path penyimpanan model .joblib")
    parser.add_argument("--vec-out", type=str, default=str(DEFAULT_VECTORIZER_FILE), help="Path penyimpanan vectorizer .joblib")
    parser.add_argument("--fig-out", type=str, default=str(DEFAULT_FIG_FILE), help="Path penyimpanan grafik confusion matrix")
    parser.add_argument("--pred-out", type=str, default=str(DEFAULT_PRED_20_FILE), help="Path penyimpanan hasil prediksi 20% uji")
    
    args = parser.parse_args()

    train_and_evaluate(
        train_path=Path(args.data),
        model_output_path=Path(args.model_out),
        vectorizer_output_path=Path(args.vec_out),
        fig_output_path=Path(args.fig_out),
        pred_20_output_path=Path(args.pred_out)
    )
