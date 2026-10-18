"""
Script: fetch_kaggle_dataset.py
Deskripsi: Mengunduh dan menyiapkan dataset nyata dari Kaggle / UCI Machine Learning Repository
Topik: 'Predict Students' Dropout and Academic Success' (Realinho et al., 2022)
Kaggle Dataset: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
UCI ML Repo: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
"""

import os
import urllib.request
import pandas as pd

def fetch_and_save_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    target_csv = os.path.join(raw_dir, "student_dropout_dataset.csv")
    
    # URL repositori data benchmark resmi (Mirror dataset Kaggle/UCI)
    dataset_url = "https://raw.githubusercontent.com/ranga4all1/student-dropout-and-success-prediction/main/data/dataset.csv"
    
    print(f"Mengunduh dataset nyata dari sumber repositori Kaggle/UCI...")
    print(f"Sumber URL: {dataset_url}")
    
    df = pd.read_csv(dataset_url, sep=",")
    
    # Standarisasi penamaan kolom (mengganti spasi dan karakter khusus agar konsisten dan ramah model ML)
    clean_columns = [
        col.strip()
           .replace(" ", "_")
           .replace("'", "")
           .replace("/", "_")
           .replace("(", "")
           .replace(")", "")
        for col in df.columns
    ]
    df.columns = clean_columns
    
    # Tambahkan kolom biner Target_Dropout untuk kemudahan klasifikasi deteksi dini
    # 1: Dropout (Kasus Kritis Peringatan Dini), 0: Non-Dropout (Graduate / Enrolled)
    df["Target_Dropout"] = (df["Target"] == "Dropout").astype(int)
    
    # Simpan dataset
    df.to_csv(target_csv, index=False)
    
    print(f"\n[OK] Dataset Kaggle nyata berhasil disimpan di: {target_csv}")
    print(f"Total Sampel Riil : {df.shape[0]} baris")
    print(f"Total Kolom/Fitur  : {df.shape[1]} kolom (termasuk Target & Target_Dropout)")
    print("\nDistribusi Target 3-Kelas Riil:")
    print(df["Target"].value_counts())
    print(df["Target"].value_counts(normalize=True).round(4) * 100)
    print("\nDistribusi Target Biner Riil (Target_Dropout):")
    print(df["Target_Dropout"].value_counts())
    print(df["Target_Dropout"].value_counts(normalize=True).round(4) * 100)

if __name__ == "__main__":
    fetch_and_save_dataset()
