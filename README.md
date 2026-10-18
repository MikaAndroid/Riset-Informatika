# 🎓 Repositori Riset Informatika: Early Warning System Retensi Mahasiswa & XAI

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.14-blue.svg?logo=python)](https://www.python.org/)
[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle%20%7C%20UCI%20Repo-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Gradient%20Boosting-green.svg)](https://en.wikipedia.org/wiki/Gradient_boosting)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20Framework-purple.svg)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Tahap-UTS%20Selesai%20%7C%20UAS%20Prepared-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Mata Kuliah**: Riset Informatika (Informatics Research)  
> **Topik Riset**: *"Implementasi Algoritma Gradient Boosting dan Explainable Artificial Intelligence (XAI) Berbasis SHAP untuk Deteksi Dini Risiko Mahasiswa Drop-Out pada Pendidikan Tinggi"*  
> *(Early Detection of Student Drop-Out Risk in Higher Education Using Gradient Boosting and SHAP-Based Explainable Artificial Intelligence)*

---

## 🗺️ Peta Struktur Repositori (Repository Architecture Map)

Repositori ini disusun secara modular memisahkan luaran **Evaluasi Tengah Semester (UTS)** dan rencana luaran **Evaluasi Akhir Semester (UAS)**:

```plaintext
Riset-Informatika/
├── 📄 README.md                        # Peta utama struktur repositori & panduan navigasi
├── 📄 requirements.txt                 # Dependensi pustaka Python
├── 📄 .gitignore                       # Filter file sementara, cache, & virtualenv
├── 📄 LICENSE                          # Lisensi open-source MIT
│
├── 📂 UTS/                             # [SELESAI] Luaran Tugas Evaluasi Tengah Semester
│   ├── 📂 01-Proposal-Penelitian/      # Proposal Akademik Lengkap 4 Bab
│   │   ├── 📄 README.md                # Ringkasan eksekutif proposal & struktur bab
│   │   └── 📄 Proposal_Penelitian.md   # Naskah proposal formal (Bab I - IV + 20 Pustaka)
│   │
│   ├── 📂 02-Dataset/                  # Dataset Empiris Riil Kaggle / UCI ML Repo
│   │   ├── 📄 README.md                # Dokumentasi sumber Kaggle, sebaran kelas, & panduan unduh
│   │   ├── 📄 student_dropout_dataset.csv # Data riil 4.424 rekam mahasiswa (36 kolom)
│   │   ├── 📄 data_dictionary.md       # Kamus data lengkap 34 variabel prediktor
│   │   ├── 📄 descriptive_statistics.csv # Statistik deskriptif data (mean, min, max, std)
│   │   └── 🐍 fetch_kaggle_dataset.py  # Script otomatis pengunduh dataset dari repositori resmi
│   │
│   ├── 📂 03-Research-Gap/             # Analisis State-of-the-Art (SOTA) & Kesenjangan Riset
│   │   ├── 📄 README.md                # Ringkasan visual 4 pilar gap riset & novelty
│   │   └── 📄 research_gap.md          # Matriks perbandingan 10 studi SOTA (2018–2024)
│   │
│   ├── 📂 04-Mind-Map/                 # Research Mind Map & Alur Metodologi
│   │   ├── 📄 README.md                # Panduan diagram Mermaid & taksonomi riset
│   │   └── 📄 research_mindmap.md      # Peta pikiran, flowchart CRISP-DM, & matriks masalah-solusi
│   │
│   └── 📂 05-Evaluation-Metric/        # Rencana Metrik Evaluasi & Uji Empiris Awal
│       ├── 📄 README.md                # Hasil komparasi model baseline vs usulan
│       ├── 📄 evaluation_metrics.md    # Formulasi matematis metrik, paradoks akurasi, & XAI
│       ├── 📄 metrics_summary.json     # Rekapitulasi nilai metrik numerik format JSON
│       ├── 🐍 eda_and_baseline.py      # Script Python pelatihan model & kalkulasi metrik
│       └── 📂 figures/                 # Grafik visualisasi hasil eksperimen
│           ├── 📊 class_distribution.png # Distribusi target 3-kelas & target biner
│           ├── 📊 confusion_matrices.png # Matriks konfusi Logistic Reg vs RF vs Gradient Boosting
│           ├── 📊 roc_pr_curves.png     # Kurva ROC-AUC dan Precision-Recall AUC
│           └── 📊 feature_importance.png# Top 10 indikator risiko pendorong drop-out (XAI)
│
└── 📂 UAS/                             # [PERSIAPAN] Ruang Lingkup Evaluasi Akhir Semester
    ├── 📂 Draft-Paper/                 # Naskah artikel ilmiah format IEEE / SINTA
    │   └── 📄 README.md                # Struktur naskah publikasi & target jurnal
    ├── 📂 Dataset-Final/               # Dataset final pasca feature engineering
    │   └── 📄 README.md                # Rencana rekayasa fitur & seleksi variabel
    ├── 📂 Kode/                        # Kode program final, tuning Optuna, & Dashboard Streamlit
    │   └── 📄 README.md                # Rencana arsitektur kode modul & web dashboard
    └── 📂 Presentasi/                  # Slide presentasi dan materi tayang sidang akhir
        └── 📄 README.md                # Garis besar struktur 15-20 slide presentasi
```

---

## 📌 Rincian Modul UTS (Evaluasi Tengah Semester)

Setiap folder di dalam direktori [`UTS/`](UTS/) telah memenuhi 5 syarat minimal mata kuliah Riset Informatika:

| No | Modul Direktori UTS | Berkas Utama | Deskripsi Konten |
|:---:|:---|:---|:---|
| **1** | [**01-Proposal-Penelitian**](UTS/01-Proposal-Penelitian/) | [`Proposal_Penelitian.md`](UTS/01-Proposal-Penelitian/Proposal_Penelitian.md) | Proposal riset akademik lengkap 4 Bab: Judul, Abstrak dwibahasa (ID & EN), Pendahuluan, Landasan Teori, Metodologi (CRISP-DM), Jadwal Kerja 6 Bulan (Gantt Chart), & 20 Referensi Bereputasi. |
| **2** | [**02-Dataset**](UTS/02-Dataset/) | [`student_dropout_dataset.csv`](UTS/02-Dataset/student_dropout_dataset.csv)<br>[`data_dictionary.md`](UTS/02-Dataset/data_dictionary.md) | Dataset empiris riil dari **Kaggle / UCI Machine Learning Repository** ($N = 4.424$ data mahasiswa, 36 kolom) beserta kamus data 34 variabel prediktor dan script pengunduh otomatis. |
| **3** | [**03-Research-Gap**](UTS/03-Research-Gap/) | [`research_gap.md`](UTS/03-Research-Gap/research_gap.md) | Matriks perbandingan 10 penelitian SOTA (2018–2024), pemetaan 4 pilar Research Gap (*Black-Box Gap, Early Horizon Gap, Imbalance Gap, & Prescriptive Gap*), dan analisis *Novelty*. |
| **4** | [**04-Mind-Map**](UTS/04-Mind-Map/) | [`research_mindmap.md`](UTS/04-Mind-Map/research_mindmap.md) | Visualisasi interaktif berbasis diagram **Mermaid**: Mind Map taksonomi riset menyeluruh, diagram alir metodologi CRISP-DM, dan tabel pemetaan masalah ke intervensi preskriptif. |
| **5** | [**05-Evaluation-Metric**](UTS/05-Evaluation-Metric/) | [`evaluation_metrics.md`](UTS/05-Evaluation-Metric/evaluation_metrics.md)<br>[`eda_and_baseline.py`](UTS/05-Evaluation-Metric/eda_and_baseline.py) | Formulasi matematis *Accuracy Paradox, Precision, Recall, Macro-F1, ROC-AUC, PR-AUC*, metrik evaluasi XAI (*Fidelity & Stability*), serta script eksekusi evaluasi empiris. |

---

## 📊 Ringkasan Hasil Uji Coba Empiris Data Kaggle ($N_{test} = 885$)

Model dilatih menggunakan data nyata dengan pemisahan *Stratified 80:20* ($N_{train} = 3.539$, $N_{test} = 885$).

| Model Evaluasi | Accuracy | Precision | Recall (Sens) | F1-Score (Bin) | Macro F1 | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | 0.8746 | 0.7912 | **0.8275** | 0.8090 | 0.8578 | 0.9290 | 0.8965 |
| **Random Forest (Bagging)** | 0.8757 | 0.8042 | 0.8099 | 0.8070 | 0.8577 | 0.9289 | **0.8990** |
| **Gradient Boosting (Proposed)** | **0.8836** | **0.8577** | 0.7641 | **0.8082** | **0.8623** | **0.9324** | 0.8989 |

> 💡 **Temuan Utama**: Algoritma usulan (**Gradient Boosting**) menghasilkan **Presisi tertinggi (85.77%)**, **Akurasi tertinggi (88.36%)**, dan **ROC-AUC tertinggi (0.9324)**. Presisi tinggi sangat krusial dalam sistem peringatan dini kampus untuk mencegah alarm palsu (*False Positives*) agar sumber daya intervensi Dosen Wali dapat dialokasikan tepat sasaran.

---

## 🚀 Panduan Menjalankan Eksperimen (Quickstart)

### 1. Kloning Repositori & Instalasi Dependensi
```bash
git clone https://github.com/USERNAME-ANDA/NAMA-REPO-ANDA.git
cd NAMA-REPO-ANDA
pip install -r requirements.txt
```

### 2. Mengunduh / Memperbarui Dataset Kaggle
```bash
python UTS/02-Dataset/fetch_kaggle_dataset.py
```

### 3. Menjalankan Pelatihan Baseline Model & Evaluasi Metrik
```bash
python UTS/05-Evaluation-Metric/eda_and_baseline.py
```
Semua metrik akan otomatis diperbarui di [`UTS/05-Evaluation-Metric/metrics_summary.json`](UTS/05-Evaluation-Metric/metrics_summary.json) dan grafik visualisasi di [`UTS/05-Evaluation-Metric/figures/`](UTS/05-Evaluation-Metric/figures/).

---

## 📅 Roadmap Menuju UAS (Evaluasi Akhir Semester)

```
[UTS SELESAI] ─────────────► [PENGEMBANGAN UAS] ─────────────► [LUARAN AKHIR]
• Proposal Riset              • Feature Engineering Rasio       • Draft Paper Publikasi
• Dataset Empiris 4.424 data  • Hyperparameter Tuning (Optuna)  • Dashboard Web Streamlit
• Pemetaan Research Gap       • Model-Agnostic SHAP Explainer   • Slide Presentasi Sidang
• Research Mind Map           • Uji Stabilitas Penjelasan
• Rencana Metrik Evaluasi     • Penyusunan Artikel Ilmiah
```

---

## 📄 Lisensi & Hak Cipta
Seluruh kode dan materi riset ini dilisensikan di bawah lisensi terbuka [MIT License](LICENSE).
