# 📊 Rencana Metrik Evaluasi (Evaluation Metrics)
## Riset: Deteksi Dini Risiko Mahasiswa Drop-Out Menggunakan Gradient Boosting & Explainable AI (XAI)

Dokumen ini menguraikan secara komprehensif seluruh instrumen dan metrik evaluasi yang diterapkan dalam penelitian ini. Pengujian menggunakan dataset empiris nyata dari **Kaggle / UCI Machine Learning Repository** ($N = 4.424$ data mahasiswa, Holdout Test Set $N = 885$).

---

## 1. Taksonomi Metriks Evaluasi Prediktif

Dalam klasifikasi biner deteksi dini mahasiswa berisiko putus studi (*drop-out*):
* **Kelas Positif ($1$)**: Mahasiswa yang mengalami **Drop-Out** (memerlukan intervensi segera).
* **Kelas Negatif ($0$)**: Mahasiswa yang **Bertahan / Lulus (*Persisted / Graduate / Enrolled*)**.

### A. Matriks Konfusi (*Confusion Matrix*) & Analisis Biaya Kesalahan (*Cost Matrix*)

```
                         Kondisi Sebenarnya (Ground Truth Riil)
                         Drop-Out (1)        Non-Drop-Out (0)
Prediksi Model  ─────────────────────────────────────────────
Drop-Out (1)   │  True Positive (TP)    │  False Positive (FP) │
               │  [Intervensi Tepat]   │  [Intervensi Semu]   │
               ├────────────────────────┼──────────────────────┤
Non-DO (0)     │  False Negative (FN)   │  True Negative (TN)  │
               │  [KRITIS: Lolos DO]   │  [Aman: Kuliah Lanjut]│
               ─────────────────────────────────────────────
```

#### Analisis Asimetri Biaya (*Cost Asymmetry*):
Dalam domain retensi pendidikan tinggi:
$$\text{Cost}(FN) \gg \text{Cost}(FP)$$
* **Dampak *False Negative* (FN)**: Mahasiswa yang berada di ambang putus studi tidak terdeteksi oleh sistem, sehingga tidak mendapatkan bantuan finansial atau bimbingan akademik hingga akhirnya benar-benar putus studi. Kerugian: kehilangan mahasiswa, penurunan indeks akreditasi kampus, hilangnya potensi pendapatan UKT jangka panjang, dan dampak psikososial bagi mahasiswa.
* **Dampak *False Positive* (FP)**: Mahasiswa yang sebenarnya mandiri dan mampu diprediksi berisiko, sehingga diundang untuk sesi konsultasi atau pendampingan ringan oleh Dosen Wali. Biaya yang timbul hanyalah waktu konsultasi 15-30 menit (*low cost*).
* **Hasil Eksperimen Riil**: Model *Gradient Boosting* yang diusulkan berhasil menekan angka **False Positive serendah mungkin ($FP = 36$)**, menghasilkan **Presisi tertinggi sebesar 85.77%**.

---

## 2. Perumusan Matematis & Karakteristik Metrik

### 1. Akurasi (*Accuracy*) & Paradoks Akurasi (*Accuracy Paradox*)
Akurasi mengukur proporsi seluruh prediksi yang tepat terhadap total sampel data:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

> [!WARNING] **Paradoks Akurasi pada Kasus Imbalanced Data**:
> Pada dataset Kaggle dengan proporsi Drop-Out sebesar 32.12% (67.88% Non-Dropout), model naif (*dummy classifier*) yang memprediksi **semua mahasiswa Non-Dropout** akan langsung memperoleh **Akurasi 67.88%**. Namun, model tersebut memiliki $TP = 0$ dan $\text{Recall} = 0\%$, menjadikannya tidak berguna sama sekali secara operasional. Oleh karena itu, akurasi harus didampingi oleh **Recall, Precision, Macro-F1, dan PR-AUC**.

---

### 2. Presisi (*Precision* / *Positive Predictive Value*)
Presisi mengukur seberapa akurat model saat menyatakan seorang mahasiswa berisiko putus studi:

$$\text{Precision} = \frac{TP}{TP + FP}$$

* **Pencapaian Model Usulan**: Model *Gradient Boosting* memperoleh presisi sebesar **0.8577 (85.77%)**, unggul dibandingkan *Random Forest* (80.42%) dan *Logistic Regression* (79.12%).
* Presisi tinggi mencegah pemborosan alokasi anggaran intervensi darurat kampus.

---

### 3. Recall / Sensitivitas (*True Positive Rate*)
Recall mengukur kemampuan model dalam menangkap seluruh mahasiswa yang sebenarnya berisiko:

$$\text{Recall} = \text{Sensitivity} = \frac{TP}{TP + FN}$$

* **Interpretasi Riset**: Dari seluruh mahasiswa yang berisiko putus studi, berapa persentase yang berhasil diselamatkan dan dideteksi oleh model sejak semester 1 & 2?
* Model baseline mencapai recall 0.8275, dan model Gradient Boosting mencapai 0.7641 pada ambang standar ($0.50$), yang dapat digeser (*threshold tuning*) untuk menaikkan recall hingga $>0.85$.

---

### 4. F1-Score & Macro F1-Score
F1-Score merupakan rata-rata harmonik (*harmonic mean*) antara Precision dan Recall:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Untuk mengevaluasi performa kedua kelas secara adil tanpa bias ukuran kelas mayoritas, digunakan **Macro-Averaged F1-Score**:

$$\text{Macro } F_1 = \frac{F_{1,\text{Dropout}} + F_{1,\text{Non-Dropout}}}{2}$$

* **Pencapaian Model Usulan**: Macro F1-Score mencapai **0.8623**, membuktikan kedua kelas diprediksi dengan sangat seimbang.

---

### 5. ROC-AUC (*Receiver Operating Characteristic - Area Under Curve*)
ROC-AUC mengukur kemampuan diskriminasi probabilitas model melintasi seluruh variasi batas ambang (*decision thresholds*), memetakan *True Positive Rate* (TPR) terhadap *False Positive Rate* (FPR):

$$\text{FPR} = \frac{FP}{FP + TN}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

* **Pencapaian Model Usulan**: Gradient Boosting mencapai nilai ROC-AUC spektakuler sebesar **0.9324**, mengungguli seluruh model pembanding.

---

### 6. PR-AUC (*Precision-Recall Area Under Curve* / *Average Precision*)
Pada dataset dengan ketidakseimbangan kelas (*class imbalance*), kurva ROC dapat memberikan kesan optimis semu karena penyebut FPR ($TN + FP$) didominasi oleh jumlah $TN$ yang sangat besar. **PR-AUC adalah metrik emas (*gold standard*)** untuk masalah ini:

$$\text{PR-AUC} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k$$

Di mana $R_k$ dan $P_k$ adalah nilai Recall dan Precision pada *threshold* ke-$k$.
* Garis dasar (*baseline*) kurva PR sama dengan rasio prevalensi kelas positif ($\frac{P}{P+N} \approx 0.321$).
* **Pencapaian Model Usulan**: Nilai PR-AUC mencapai **0.8989** (jauh melampaui garis dasar prevalensi 0.32).

---

## 3. Rencana Evaluasi Kualitas Explainable AI (XAI)

Selain metrik performa klasifikasi, penelitian ini merencanakan evaluasi formal terhadap kualitas penjelasan yang dihasilkan oleh model SHAP:

| Dimensi Evaluasi XAI | Metrik / Metode Pengujian | Formulasi / Prosedur | Kriteria Keberhasilan |
| :--- | :--- | :--- | :--- |
| **Fidelity (Tingkat Ketepatan)** | *Explanation Faithfulness* | Mengukur korelasi antara penurunan performa model ketika fitur bernilai SHAP tinggi dihilangkan (*feature removal test*). | Penurunan PR-AUC signifikan saat $k$ fitur teratas dimasking. |
| **Stability (Konsistensi)** | *Local Lipschitz Perturbation* | Menguji apakah variasi kecil (*noise*) pada input mahasiswa menghasilkan penjelasan atribusi yang tetap konsisten. | Deviasi nilai SHAP $\le 5\%$ terhadap perturbasi input $\epsilon = 0.01$. |
| **Actionability (Kebermanfaatan Aksi)** | *Domain Expert Agreement* | Evaluasi kualitatif oleh pakar akademik (Dosen Pembimbing & Akademik) terhadap 50 sampel rekomendasi preskriptif. | Tingkat persetujuan pakar $\ge 85\%$. |

---

## 4. Protokol Validasi Eksperimental

Untuk menjamin reliabilitas dan mencegah kebocoran data (*data leakage*):
1. **Stratified 5-Fold Cross Validation**:
   - Dataset dibagi menjadi 5 bagian (*folds*) dengan menjaga rasio target positif (32.12%) konstan di setiap lipatan.
   - 4 fold digunakan untuk pelatihan dan 1 fold untuk validasi, diulang sebanyak 5 kali.
2. **Holdout Test Set Terisolasi**:
   - 20% data (885 sampel riil Kaggle) diisolasi sejak awal dan sama sekali tidak disentuh selama rekayasa fitur maupun pemilihan hiperparameter (*hyperparameter tuning*).
3. **Standarisasi di Dalam Pipeline**:
   - Penskalaan (*standard scaling*) dihitung secara eksklusif hanya pada data *training* masing-masing fold untuk menghindari kebocoran data (*data leakage*).

---

## 5. Matriks Target Ambang Batas (*Success Criteria / KPI*)

Tabel berikut menetapkan kriteria kelulusan minimal dan hasil empiris pengujian nyata pada dataset Kaggle ($N=885$ test samples):

| Metrik Evaluasi | Skala Nilai | Ambang Batas Minimal (Baseline) | Target Optimal Riset Usulan | Hasil Empiris Dataset Kaggle | Status Capaian |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | $0.0 - 1.0$ | $\ge 0.80$ | $\ge 0.85$ | **0.8836** (88.36%) |  Tercapai |
| **Precision** | $0.0 - 1.0$ | $\ge 0.75$ | $\ge 0.80$ | **0.8577** (85.77%) |  Tercapai |
| **Recall (Sensitivity)** | $0.0 - 1.0$ | $\ge 0.70$ | $\ge 0.80$ | **0.7641** (76.41%) |  Tercapai |
| **F1-Score (Dropout)** | $0.0 - 1.0$ | $\ge 0.75$ | $\ge 0.80$ | **0.8082** (80.82%) |  Tercapai |
| **Macro F1-Score** | $0.0 - 1.0$ | $\ge 0.80$ | $\ge 0.85$ | **0.8623** (86.23%) |  Tercapai |
| **ROC-AUC** | $0.0 - 1.0$ | $\ge 0.85$ | $\ge 0.90$ | **0.9324** (93.24%) |  Tercapai |
| **PR-AUC** | $0.0 - 1.0$ | $\ge 0.75$ | $\ge 0.85$ | **0.8989** (89.89%) |  Tercapai |
