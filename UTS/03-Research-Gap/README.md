# 🔬 03 - Research Gap & Analisis State-of-the-Art (SOTA)
## Pemetaan Posisi Riset dan Kebaruan (Novelty)

Folder ini berisi analisis mendalam mengenai tinjauan pustaka terkini, perbandingan matriks penelitian sebelumnya, identifikasi celah riset (*research gap*), serta kebaruan (*novelty*) yang diusulkan.

---

### 📄 Berkas Utama
* **Dokumen Lengkap**: [`research_gap.md`](research_gap.md)

---

### 📌 Ringkasan 4 Celah Riset Utama (The 4 Research Gaps)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        IDENTIFIKASI RESEARCH GAP                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌─────────────────┬─────────────┴───────┬──────────────────┐
    ▼                 ▼                     ▼                  ▼
┌──────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ 1. GAP METODE│ │ 2. GAP TEMPORAL│ │ 3. GAP DATA    │ │ 4. GAP AKSI    │
│  & XAI       │ │  (EARLY-STAGE) │ │    IMBALANCE   │ │  PRESKRIPTIF   │
│ Model akurat │ │ Mayoritas riset│ │ Akurasi semu   │ │ Output sebatas │
│ tapi black-  │ │ prediksi di S4-│ │ tinggi padahal │ │ label 'DO', tak│
│ box; nihil   │ │ S6; intervensi │ │ Recall kasus DO│ │ ada rekomendasi│
│ transparansi │ │ telat dilakukan│ │ sangat rendah  │ │ solusi spesifik│
└──────────────┘ └────────────────┘ └────────────────┘ └────────────────┘
```

1. **Methodological & Interpretability Gap (Black-Box Problem)**: Model prediksi konvensional tidak memberikan penjelasan kausalitas mengapa seorang mahasiswa dinyatakan berisiko. Riset ini menyematkan **Explainable AI berbasis SHAP** untuk transparansi global dan lokal.
2. **Temporal Horizon & Early-Warning Gap (Deteksi Terlalu Larut)**: Mayoritas studi literatur menggunakan data semester 4 hingga semester 6 saat mahasiswa sudah terlanjur terlambat diselamatkan. Riset ini membatasi observasi ketat pada **Tahun Pertama (Semester 1 dan Semester 2)**.
3. **Data Imbalance & Evaluation Metric Gap (Ilusi Akurasi Semu)**: Penilaian menggunakan akurasi semata menyesatkan pada data timpang. Riset ini berfokus pada **PR-AUC**, **Recall**, dan **Macro F1-Score**.
4. **Actionability & Prescriptive Intervention Gap (Ketiadaan Rekomendasi Solusi)**: Model prediktif yang ada hanya mengeluarkan probabilitas tanpa rekomendasi aksi. Riset ini memetakan nilai SHAP ke dalam katalog intervensi akademik vs finansial bagi Dosen Wali.

---

### 🌟 Kebaruan Riset (*Novelty*)
* Integrasi *Gradient Boosting* + dekomposisi SHAP ganda (*Global Feature Attribution* dan *Local Decision Waterfall*).
* Sistem deteksi dini terkalibrasi khusus tahun pertama perkuliahan dengan metrik *PR-AUC*.
* Pemetaan langsung dari nilai matematis *Shapley* ke dalam intervensi preskriptif operasional kampus.
