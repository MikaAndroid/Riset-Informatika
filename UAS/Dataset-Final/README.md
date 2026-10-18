# 💾 UAS: Dataset Final & Feature Engineering
## Data Preprocessing, Feature Selection, dan Dataset Akhir

Folder ini disiapkan untuk menampung **Dataset Akhir (Final)** yang telah melalui tahap rekayasa fitur (*feature engineering*), seleksi fitur optimal (*feature selection*), dan penanganan ketidakseimbangan kelas (*advanced class balancing*).

---

### 🎯 Rencana Roadmap Pengolahan Data Final
1. **Feature Engineering**:
   * Rasio kelulusan SKS Semester 1 dan Semester 2: $\frac{\text{SKS Lulus}}{\text{SKS Enrolled}}$.
   * Rasio efisiensi evaluasi: $\frac{\text{SKS Lulus}}{\text{Evaluasi Diikuti}}$.
   * Indikator deviasi nilai rata-rata terhadap batas kelulusan standar.
2. **Feature Selection & Multicollinearity Check**:
   * Analisis *Variance Inflation Factor* (VIF) untuk menghilangkan redundansi fitur.
   * Seleksi fitur berbasis SHAP rank correlation.
3. **Penyimpanan Format Data Final**:
   * `train_final.csv` & `test_final.csv`.
   * Metadata & transformasi scaler tersimpan (`scaler.joblib`).
