# 📊 Rencana Metrik Evaluasi (Evaluation Metrics)
## Riset: Deteksi Dini Risiko Mahasiswa Drop-Out Menggunakan Gradient Boosting & Explainable AI (XAI)

Dokumen ini menguraikan secara komprehensif seluruh instrumen dan metrik evaluasi yang direncanakan dalam penelitian ini. Setiap metrik dijelaskan mulai dari perumusan matematis, justifikasi teoritis pemilihan metrik, analisis matriks kerugian (*cost-sensitive trade-off*), hingga metrik evaluasi khusus untuk lapisan kejelasan model (*Explainable AI*).

---

## 1. Taksonomi Metriks Evaluasi Prediktif

Dalam klasifikasi biner deteksi dini mahasiswa berisiko putus studi (*drop-out*):
* **Kelas Positif ($1$)**: Mahasiswa yang mengalami **Drop-Out** (memerlukan intervensi segera).
* **Kelas Negatif ($0$)**: Mahasiswa yang **Bertahan / Lulus (*Persisted / Graduate*)**.

### A. Matriks Konfusi (*Confusion Matrix*) & Analisis Biaya Kesalahan (*Cost Matrix*)

```
                         Kondisi Sebenarnya (Ground Truth)
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
* **Kesimpulan Desain**: Model harus dioptimalkan untuk memiliki **Recall (Sensitivity)** setinggi mungkin, tanpa mengorbankan **Precision** secara drastis.

---

## 2. Perumusan Matematis & Karakteristik Metrik

### 1. Akurasi (*Accuracy*) & Paradoks Akurasi (*Accuracy Paradox*)
Akurasi mengukur proporsi seluruh prediksi yang tepat terhadap total sampel data:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

> [!WARNING] **Paradoks Akurasi pada Kasus Imbalanced Data**:
> Pada dataset dengan proporsi Drop-Out sebesar 16.1% (seperti pada dataset riset ini di mana 83.9% mahasiswa Non-Dropout), model naif (*dummy classifier*) yang memprediksi **semua mahasiswa Non-Dropout** akan langsung memperoleh **Akurasi 83.9%**. Namun, model tersebut memiliki $TP = 0$ dan $\text{Recall} = 0\%$, menjadikannya sama sekali tidak berguna secara operasional. Oleh karena itu, **akurasi tidak dapat dijadikan metrik tunggal atau metrik utama**.

---

### 2. Presisi (*Precision* / *Positive Predictive Value*)
Presisi mengukur seberapa akurat model saat menyatakan seorang mahasiswa berisiko putus studi:

$$\text{Precision} = \frac{TP}{TP + FP}$$

* **Interpretasi Riset**: Dari seluruh mahasiswa yang dibunyikan alarm peringatan dininya oleh model, berapa persentase yang benar-benar membutuhkan intervensi?
* Presisi tinggi mencegah pemborosan alokasi anggaran intervensi darurat kampus.

---

### 3. Recall / Sensitivitas (*True Positive Rate*)
Recall mengukur kemampuan model dalam menangkap seluruh mahasiswa yang sebenarnya berisiko:

$$\text{Recall} = \text{Sensitivity} = \frac{TP}{TP + FN}$$

* **Interpretasi Riset**: Dari seluruh mahasiswa yang berisiko putus studi, berapa persentase yang berhasil diselamatkan dan dideteksi oleh model sejak semester 1 & 2?
* Metrik ini merupakan **metrik prioritas utama** dalam sistem peringatan dini kampus.

---

### 4. F1-Score & Macro F1-Score
F1-Score merupakan rata-rata harmonik (*harmonic mean*) antara Precision dan Recall:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Untuk mengevaluasi performa kedua kelas secara adil tanpa bias ukuran kelas mayoritas, digunakan **Macro-Averaged F1-Score**:

$$\text{Macro } F_1 = \frac{F_{1,\text{Dropout}} + F_{1,\text{Non-Dropout}}}{2}$$

---

### 5. ROC-AUC (*Receiver Operating Characteristic - Area Under Curve*)
ROC-AUC mengukur kemampuan diskriminasi probabilitas model melintasi seluruh variasi batas ambang (*decision thresholds*), memetakan *True Positive Rate* (TPR) terhadap *False Positive Rate* (FPR):

$$\text{FPR} = \frac{FP}{FP + TN}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

* Nilai 0.50 setara tebakan acak (*random guess*), sedangkan 1.00 menandakan pemisahan kelas yang sempurna.

---

### 6. PR-AUC (*Precision-Recall Area Under Curve* / *Average Precision*)
Pada dataset dengan ketidakseimbangan kelas (*class imbalance*), kurva ROC dapat memberikan kesan optimis semu karena penyebut FPR ($TN + FP$) didominasi oleh jumlah $TN$ yang sangat besar. **PR-AUC adalah metrik emas (*gold standard*)** untuk masalah ini:

$$\text{PR-AUC} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k$$

Di mana $R_k$ dan $P_k$ adalah nilai Recall dan Precision pada *threshold* ke-$k$.
* Garis dasar (*baseline*) kurva ROC selalu bernilai 0.50, namun garis dasar kurva PR sama dengan rasio prevalensi kelas positif ($\frac{P}{P+N} \approx 0.161$).
* Model yang menghasilkan PR-AUC $> 0.85$ membuktikan kapabilitas diskriminasi yang luar biasa pada kelas minoritas berisiko.

---

## 3. Rencana Evaluasi Kualitas Explainable AI (XAI)

Selain metrik performa klasifikasi, penelitian ini merencanakan evaluasi formal terhadap kualitas penjelasan yang dihasilkan oleh model SHAP:

| Dimensi Evaluasi XAI | Metrik / Metode Pengujian | Formulasi / Prosedur | Kriteria Keberhasilan |
| :--- | :--- | :--- | :--- |
| **Fidelity (Tingkat Ketepatan)** | *Explanation Faithfulness* | Mengukur korelasi antara penurunan akurasi model ketika fitur bernilai SHAP tinggi dihilangkan (*feature removal test*). | Penurunan performa signifikan saat $k$ fitur teratas dimasking. |
| **Stability (Konsistensi)** | *Local Lipschitz Perturbation* | Menguji apakah variasi kecil (*noise*) pada input mahasiswa menghasilkan penjelasan atribusi yang tetap konsisten. | Deviasi nilai SHAP $\le 5\%$ terhadap perturbasi input $\epsilon = 0.01$. |
| **Actionability (Kebermanfaatan Aksi)** | *Domain Expert Agreement* | Evaluasi kualitatif oleh pakar akademik (Dosen Pembimbing & Akademik) terhadap 50 sampel rekomendasi preskriptif. | Tingkat persetujuan pakar $\ge 85\%$. |

---

## 4. Protokol Validasi Eksperimental

Untuk menjamin reliabilitas dan mencegah kebocoran data (*data leakage*):
1. **Stratified 5-Fold Cross Validation**:
   - Dataset dibagi menjadi 5 bagian (*folds*) dengan menjaga rasio target positif (16.1%) konstan di setiap lipatan.
   - 4 fold digunakan untuk pelatihan dan 1 fold untuk validasi, diulang sebanyak 5 kali.
2. **Holdout Test Set Terisolasi**:
   - 20% data (300 sampel) diisolasi sejak awal dan sama sekali tidak disentuh selama rekayasa fitur maupun pemilihan hiperparameter (*hyperparameter tuning*).
3. **Standarisasi di Dalam Pipeline**:
   - Proses penskalaan (*standard scaling*) dihitung secara eksklusif hanya pada data *training* masing-masing fold untuk menghindari kebocoran statistik mean dan varians ke data evaluasi.

---

## 5. Matriks Target Ambang Batas (*Success Criteria / KPI*)

Tabel berikut menetapkan kriteria kelulusan minimal dan target optimal yang ingin dicapai dalam proposal riset ini:

| Metrik Evaluasi | Skala Nilai | Ambang Batas Minimal (Baseline) | Target Optimal Riset Usulan | Hasil Pengujian Awal (Small Test) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall (Sensitivity)** | $0.0 - 1.0$ | $\ge 0.75$ | $\ge 0.85$ | **0.8542** |  Tercapai |
| **Precision** | $0.0 - 1.0$ | $\ge 0.70$ | $\ge 0.80$ | **0.8913** |  Tercapai |
| **F1-Score (Dropout)** | $0.0 - 1.0$ | $\ge 0.72$ | $\ge 0.82$ | **0.8723** |  Tercapai |
| **Macro F1-Score** | $0.0 - 1.0$ | $\ge 0.80$ | $\ge 0.90$ | **0.9243** |  Tercapai |
| **ROC-AUC** | $0.0 - 1.0$ | $\ge 0.85$ | $\ge 0.95$ | **0.9909** |  Tercapai |
| **PR-AUC** | $0.0 - 1.0$ | $\ge 0.75$ | $\ge 0.90$ | **0.9538** |  Tercapai |
| **Overall Accuracy** | $0.0 - 1.0$ | $\ge 0.85$ | $\ge 0.92$ | **0.9600** |  Tercapai |

> Catatan: Pengujian awal pada test set membuktikan bahwa algoritma *Gradient Boosting* yang diusulkan berhasil melampaui seluruh target ambang batas minimal, khususnya pada PR-AUC (0.9538) dan Precision (0.8913).
