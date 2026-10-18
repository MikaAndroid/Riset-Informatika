# 🧠 04 - Research Mind Map
## Peta Pemikiran Riset & Alur Metodologi Penelitian

Folder ini memuat visualisasi peta pemikiran komprehensif, diagram alur metodologi (*workflow*), serta matriks pemetaan masalah-metrik-intervensi berbasis diagram **Mermaid**.

---

### 📄 Berkas Utama
* **Dokumen Lengkap**: [`research_mindmap.md`](research_mindmap.md)

---

### 🗺️ Peta Pemikiran Riset Utama (*Research Mind Map*)

```mermaid
mindmap
  root((Deteksi Dini Drop-Out & XAI))
    1. Problem Space (Urgensi)
      Tingginya Attrition Rate
        Beban Finansial Kampus
        Dampak Akreditasi Prodi
        Kerugian Waktu Mahasiswa
      Kelemahan Model Saat Ini
        Black-Box: Tanpa Alasan Kausal
        Prediksi Terlambat S4-S6
        Bias Akurasi Semu Imbalanced Data
    2. Theoretical Foundation
      Educational Data Mining (EDM)
        Learning Analytics
        Student Retention Models
      Gradient Boosting Frameworks
        XGBoost
        LightGBM
        CatBoost
      Cooperative Game Theory
        Shapley Values
        Additive Feature Attribution (SHAP)
    3. Data Pipeline & Features
      Profil Demografi
        Gender, Usia, Status Perkawinan, Asal Daerah (Displaced)
      Latar Belakang Sosio-Ekonomi
        Status Beasiswa, Keteraturan UKT/SPP, Tunggakan (Debtor)
      Rekam Akademik Awal
        Semester 1: SKS Lulus, Nilai Rata-rata, Frekuensi Evaluasi
        Semester 2: SKS Lulus, Nilai Rata-rata, Frekuensi Evaluasi
      Konteks Makro-Ekonomi
        Tingkat Pengangguran, Inflasi, GDP
    4. Experimental & Validation
      Data Preprocessing
        Imputasi & Skalasi Standar
        Penanganan Imbalance: Cost-Sensitive / Class Weights
      Validation Scheme
        Stratified 5-Fold Cross Validation
        No Data Leakage Protocol
      Baseline vs Proposed Models
        Logistic Regression (Linear Baseline)
        Random Forest (Bagging Baseline)
        Gradient Boosting (Proposed)
    5. Evaluation Metrics
      Imbalance-Resilient Metrics
        PR-AUC (Precision-Recall AUC)
        Macro F1-Score & Binary F1
        Recall (Sensitivity Minimasi False Negatives)
      Threshold Analysis
        Confusion Matrix (TP, TN, FP, FN)
        Cost of Misclassification
    6. Explainability Layer (XAI)
      Global Explanations
        Summary Plot: Fitur Paling Berpengaruh Kampus
        Feature Interaction Effects
      Local Explanations
        Waterfall Plot: Alasan Mahasiswa Spesifik
        Force Plot: Pendorong vs Penahan Risiko
    7. Decision Support & Action
      Prescriptive Interventions
        Masalah UKT: Skema Cicilan / Beasiswa Darurat
        Masalah Akademik: Remedial & Mentoring Tutor Sebaya
      Early Warning System Dashboard
        Sistem Peringatan Dosen Wali
        Monitoring Terpadu Semester 1-2
```

---

### 🔄 Diagram Alur Kerja Riset (*End-to-End Workflow*)

```mermaid
flowchart TD
    classDef startend fill:#1b9e77,stroke:#000,stroke-width:1.5px,color:#fff;
    classDef step fill:#f0f4f8,stroke:#2b5c8f,stroke-width:1.5px,color:#111;
    classDef decision fill:#fff2cc,stroke:#d6b656,stroke-width:1.5px,color:#000;
    classDef xai fill:#e1d5e7,stroke:#9673a6,stroke-width:1.5px,color:#000;
    classDef output fill:#d5e8d4,stroke:#82b366,stroke-width:1.5px,color:#000;

    Start([Mulai: Identifikasi Masalah Drop-Out]) :::startend --> DataPrep[Pengumpulan Data Kaggle/UCI:<br>4.424 Rekam Mahasiswa] :::step
    DataPrep --> Preprocessing[Data Preprocessing:<br>Pembersihan, Verifikasi, Standarisasi] :::step
    Preprocessing --> Split[Stratified Splitting:<br>80% Train / 20% Test] :::step
    
    Split --> TrainBaseline[Latih Baseline:<br>Logistic Regression & Random Forest] :::step
    Split --> TrainGB[Latih Model Usulan:<br>Gradient Boosting dengan Class Weights] :::step
    
    TrainBaseline --> EvalComp[Evaluasi Komparasi Model:<br>PR-AUC, ROC-AUC, Recall, Precision, Macro-F1] :::decision
    TrainGB --> EvalComp
    
    EvalComp --> XAILayer[Lapisan Explainable AI: SHAP Explainer Engine] :::xai
    XAILayer --> GlobalSHAP[Analisis Global: Fitur Risiko Kampus] :::xai
    XAILayer --> LocalSHAP[Analisis Lokal: Waterfall Plot Per Mahasiswa] :::xai
    
    GlobalSHAP --> Prescriptive[Pemetaan Intervensi Preskriptif:<br>Kategori Finansial vs Kategori Akademik] :::output
    LocalSHAP --> Prescriptive
    Prescriptive --> Deployment([Integrasi Early Warning Dashboard Dosen Wali]) :::startend
```
