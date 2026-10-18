# 💻 UAS: Kode Sumber & Sistem Prediksi Final
## Implementasi Algoritma Lengkap, Hyperparameter Tuning, & XAI Dashboard

Folder ini disiapkan untuk seluruh **Kode Sumber (Source Code)** final dari sistem deteksi dini dan penjelasan XAI.

---

### 🎯 Rencana Modul Kode yang Akan Dikembangkan
1. **`preprocessing.py`**: Pipeline pembersihan, standarisasi, dan rekayasa fitur otomatis.
2. **`train_and_tune.py`**: Script *hyperparameter optimization* menggunakan kerangka kerja modern (**Optuna**) untuk XGBoost, LightGBM, CatBoost.
3. **`xai_explainer.py`**: Modul penghitung nilai SHAP lokal (*waterfall plot*) dan global (*beeswarm plot*), serta kalkulasi indeks kestabilan (*Lipschitz stability*).
4. **`app.py` / `dashboard/`**: Prototipe aplikasi antarmuka web interaktif (*Streamlit*) untuk simulasi prediksi risiko mahasiswa dan visualisasi alasan bagi Dosen Pembimbing Akademik.
5. **`notebooks/`**: Jupyter Notebook terpadu untuk dokumentasi *step-by-step* yang dapat dijalankan di Google Colab / Kaggle Notebooks.
