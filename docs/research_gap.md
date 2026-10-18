# 🔬 Research Gap & Analisis State-of-the-Art (SOTA)
## Riset: Deteksi Dini Risiko Mahasiswa Drop-Out Menggunakan Gradient Boosting & Explainable Artificial Intelligence (XAI) Berbasis SHAP

---

## 1. Latar Belakang & Urgensi Pemetaan Celah Riset
Tingkat putus studi (*drop-out*) pada pendidikan tinggi merupakan tantangan global yang merugikan mahasiswa secara personal dan membebani institusi secara finansial maupun reputasi akreditasi. Dalam satu dekade terakhir, *Educational Data Mining* (EDM) telah banyak menggunakan algoritma *Machine Learning* untuk memprediksi mahasiswa berisiko. Namun, mayoritas model prediktif yang diusulkan dalam literatur masih berfungsi sebagai **"Black-Box"** (kotak hitam) yang hanya menghasilkan probabilitas tanpa penjelasan kausal yang dapat ditindaklanjuti (*actionable insights*) oleh dosen wali (*academic advisor*).

Dokumen ini memetakan posisi penelitian terhadap literatur ilmiah terkini (*State-of-the-Art*), mengidentifikasi 4 celah riset utama (*Research Gaps*), dan menegaskan kebaruan (*novelty*) serta kontribusi yang ditawarkan.

---

## 2. Matriks Kajian Pustaka & State-of-the-Art (SOTA)

Tabel berikut merangkum 10 penelitian terkemuka dalam domain prediksi retensi mahasiswa:

| No | Peneliti & Tahun | Metode / Algoritma | Dataset & Cakupan | Metrik Kunci | Kelebihan | Keterbatasan & Celah Riset (*Gap*) |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | **Realinho et al. (2022)**<br>*(Nature Scientific Data)* | Random Forest, SVM, MLP | 4.424 mahasiswa (Polytechnic of Portalegre) | Accuracy: 77.2% | Mempublikasikan dataset benchmark standar internasional. | Fokus hanya pada komparasi akurasi standar; tidak ada interpretasi berbasis XAI dan tidak mengeksplorasi ketidakseimbangan kelas (*class imbalance*). |
| 2 | **Agrusti et al. (2020)**<br>*(Computers & Education)* | Decision Tree, Logistic Regression | Data historis 2 universitas di Italia | F1-Score: 0.72 | Model mudah diinterpretasikan secara manual (*white-box*). | Akurasi prediktif rendah; tidak mampu menangkap relasi non-linear yang kompleks antar variabel sosio-ekonomi. |
| 3 | **Albreiki et al. (2021)**<br>*(IEEE Access)* | XGBoost, AdaBoost, Naïve Bayes | Data universitas di Timur Tengah (3.000 mahasiswa) | Accuracy: 84.6% | Algoritma *ensemble* terbukti mengungguli model tunggal. | Prediksi dilakukan menggunakan data multi-semester hingga tahun ke-3 (terlambat untuk intervensi preventif). |
| 4 | **Bhat et al. (2022)**<br>*(IEEE TLT)* | Deep Neural Networks (DNN) & LSTM | Data interaksi LMS (Clickstream data) | ROC-AUC: 0.88 | Akurasi sekuensial tinggi pada platform daring. | *Black-box* total; sangat bergantung pada log digital LMS yang tidak tersedia seragam di semua perguruan tinggi konvensional. |
| 5 | **Ortiz-Lozano et al. (2020)**<br>*(Higher Education)* | Random Forest | 8.200 mahasiswa Spanyol | Precision: 0.69<br>Recall: 0.64 | Menyoroti faktor ekonomi sebagai prediktor penting. | Tidak menangani *trade-off* False Negative (mahasiswa DO yang terlewat diprediksi aman). |
| 6 | **Rastrollo-Guerrero et al. (2020)**<br>*(Applied Sciences)* | Systematic Review (70+ studi EDM) | Meta-analisis literatur global | - | Memetakan tren algoritma EDM 2010–2020. | Menemukan bahwa <12% riset EDM yang mengintegrasikan teknik penjelasan model (*explainability*). |
| 7 | **Baranyi et al. (2023)**<br>*(Expert Systems with Applications)* | CatBoost & SHAP dasar | Data universitas teknik Eropa Tengah | Accuracy: 81.0% | Mulai memanfaatkan SHAP untuk analisis variabel global. | Penjelasan hanya sebatas ringkasan agregat kampus (*global summary*), belum merumuskan intervensi preskriptif di level individu mahasiswa (*local explanation*). |
| 8 | **Tanuar et al. (2022)**<br>*(Jurnal Nasional SINTA 2)* | K-Nearest Neighbor (KNN) | 450 mahasiswa PTS Indonesia | Accuracy: 78.5% | Relevan dengan konteks perguruan tinggi lokal Indonesia. | Dataset sangat kecil, fitur terbatas hanya pada IPK dan SKS, tanpa analisis ketidakseimbangan kelas dan tanpa XAI. |
| 9 | **Martins et al. (2021)**<br>*(Data in Brief)* | LightGBM dengan SMOTE | 3.500 mahasiswa politeknik | F1-Macro: 0.76 | Mengatasi ketidakseimbangan kelas menggunakan oversampling. | Tidak menganalisis stabilitas nilai SHAP pasca-SMOTE (*synthetic sampling bias*). |
| 10 | **Stupariu et al. (2024)**<br>*(Int. J. of Educational Tech)* | Ensemble Stacking (RF+XGB) | 2.100 mahasiswa | PR-AUC: 0.71 | Menggunakan PR-AUC sebagai metrik imbalanced data. | Kurang membedah dampak finansial (SPP/UKT macet) sebagai fitur dinamis pada semester 1 & 2. |

---

## 3. Identifikasi Celah Riset Utama (The 4 Research Gaps)

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

### Gap 1: *Methodological & Interpretability Gap* (Masalah Kotak Hitam)
* **Kondisi Saat Ini**: Model *state-of-the-art* seperti Gradient Boosting (XGBoost, LightGBM, CatBoost) atau Neural Networks terbukti memiliki akurasi superior dibandingkan model linear (Logistic Regression). Namun, model tersebut bersifat *black-box*.
* **Celah yang Ada**: Pengambil keputusan akademik (Ketua Program Studi dan Dosen Wali) enggan mengadopsi rekomendasi AI jika tidak memahami *mengapa* seorang mahasiswa diklasifikasikan berisiko putus studi.
* **Solusi Riset Ini**: Mengintegrasikan kerangka **Explainable AI (XAI)** menggunakan **SHAP (SHapley Additive exPlanations)** yang berakar pada *Cooperative Game Theory*. Model tidak hanya memprediksi risiko, namun menghitung kontribusi marjinal pasti setiap fitur (akademik, demografi, finansial) baik secara global maupun lokal per mahasiswa.

### Gap 2: *Temporal Horizon & Early-Warning Gap* (Deteksi Terlalu Larut)
* **Kondisi Saat Ini**: Banyak riset literatur mengevaluasi mahasiswa dengan menyertakan fitur rekam jejak akademik hingga semester 4, 5, atau 6.
* **Celah yang Ada**: Ketika mahasiswa sudah berada di semester lanjut, akumulasi kegagalan akademik dan tunggakan finansial sudah terlanjur parah, sehingga tingkat keberhasilan intervensi (*retention intervention*) sangat rendah.
* **Solusi Riset Ini**: Membatasi horizon observasi ketat hanya pada **Tahun Pertama (Semester 1 dan Semester 2)**. Hal ini memastikan sistem berfungsi sebagai *Early-Warning System* (EWS) sejati yang memberikan jendela waktu intervensi maksimal bagi pihak kampus.

### Gap 3: *Data Imbalance & Evaluation Metric Gap* (Ilusi Akurasi Semu)
* **Kondisi Saat Ini**: Proporsi mahasiswa putus studi secara alamiah bersifat minoritas (biasanya 10%–25% dari total populasi mahasiswa). Banyak penelitian terdahulu hanya melaporkan *Overall Accuracy*.
* **Celah yang Ada**: Model naif yang memprediksi seluruh mahasiswa "Aman / Non-Dropout" dapat dengan mudah mencapai akurasi 85%, namun memiliki **Recall 0%** pada kelas kritis (*False Negative* tinggi). Ini sangat fatal karena mahasiswa yang benar-benar membutuhkan bantuan gagal terdeteksi.
* **Solusi Riset Ini**: Menerapkan metrik evaluasi yang kebal terhadap *class imbalance*, yaitu **Precision-Recall AUC (PR-AUC)**, **Macro F1-Score**, dan **Recall (Sensitivity)**, dipadukan dengan teknik penyeimbangan bobot kelas (*cost-sensitive learning*).

### Gap 4: *Actionability & Prescriptive Intervention Gap* (Ketiadaan Rekomendasi Konkret)
* **Kondisi Saat Ini**: Output dari model prediktif konvensional hanya berupa label biner `[0 = Tidak DO, 1 = DO]` atau skor probabilitas `0.87`.
* **Celah yang Ada**: Label tersebut tidak memberikan petunjuk tindakan apa yang harus diambil kampus: Apakah mahasiswa tersebut butuh keringanan finansial? Atau pendampingan tutor sebaya pada mata kuliah tertentu?
* **Solusi Riset Ini**: Memanfaatkan nilai *local SHAP decomposition* untuk memetakan akar penyebab dominan per mahasiswa secara otomatis ke dalam katalog intervensi:
  - Jika beban terbesar berasal dari `Tuition_Fees_Up_To_Date` & `Debtor` $\rightarrow$ Rekomendasi: Skema cicilan UKT / Beasiswa darurat.
  - Jika beban terbesar berasal dari `Curricular_Units_1st_Sem_Approved` $\rightarrow$ Rekomendasi: *Mentoring* akademik & remedial mata kuliah prasyarat.

---

## 4. Kebaruan Penelitian (*Novelty*) & Kontribusi Ilmiah

Berdasarkan pemetaan celah di atas, kebaruan dan kontribusi orisinal dari penelitian ini dirangkum sebagai berikut:

1. **Framework Prediksi Terpadu yang Terdiversifikasi dan Terinterpretasi (*Interpretable EDM Framework*)**:
   Menggabungkan ketangguhan prediktif algoritma *Gradient Boosting* dengan transparansi tingkat instansiasi (*instance-level transparency*) melalui dekomposisi SHAP ganda (*Global Feature Attribution* dan *Local Decision Waterfall*).
2. **Formulasi Protokol Deteksi Dini Berbasis Data Semester 1-2**:
   Membuktikan secara empiris bahwa kombinasi indikator keberhasilan semester 1-2, ketaatan pembayaran biaya kuliah, dan status penerima beasiswa mampu menghasilkan *PR-AUC* dan *Recall* yang tinggi tanpa harus menunggu rekam jejak tahun ke-3.
3. **Analisis Mitigasi Bias Evaluasi Imbalanced Dataset**:
   Memberikan pemodelan komparatif yang ketat antara metrik *Overall Accuracy* vs *PR-AUC* dan *Macro-F1*, memperlihatkan secara gamblang pergeseran batas ambang (*decision threshold*) optimal untuk meminimalkan *Cost of False Negatives* di lingkungan institusi pendidikan tinggi.
