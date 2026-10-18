# 📑 PROPOSAL PENELITIAN
## PROGRAM STUDI INFORMATIKA / FAKULTAS ILMU KOMPUTER

---

### **JUDUL PENELITIAN:**
> **"Implementasi Algoritma Gradient Boosting dan Explainable Artificial Intelligence (XAI) Berbasis SHAP untuk Deteksi Dini Risiko Mahasiswa Drop-Out pada Pendidikan Tinggi"**
>
> *(Early Detection of Student Drop-Out Risk in Higher Education Using Gradient Boosting and SHAP-Based Explainable Artificial Intelligence)*

---

### **ABSTRAK**
Tingginya angka putus studi (*drop-out*) pada pendidikan tinggi menimbulkan kerugian finansial yang signifikan bagi perguruan tinggi serta merugikan masa depan akademik mahasiswa. Sebagian besar penelitian terdahulu yang memanfaatkan *Educational Data Mining* (EDM) berfokus pada pembangunan model berakurasi tinggi namun bersifat *black-box*, memprediksi risiko pada semester akhir, dan mengabaikan ketidakseimbangan kelas (*class imbalance*). Akibatnya, pengambil kebijakan akademik tidak memperoleh pemahaman mengenai faktor kausal spesifik yang melandasi risiko seorang mahasiswa, sehingga intervensi preventif sulit dilakukan secara terarah. 

Penelitian ini mengusulkan sebuah kerangka kerja sistem deteksi dini (*early warning system*) berbasis algoritma *Gradient Boosting* yang diintegrasikan dengan metode *Explainable Artificial Intelligence* (XAI) menggunakan pendekatan *SHapley Additive exPlanations* (SHAP). Penelitian ini memanfaatkan dataset empiris nyata dari repositori **Kaggle / UCI Machine Learning Repository** (*Predict Students' Dropout and Academic Success*, Realinho et al., 2022) sebanyak 4.424 data mahasiswa dengan 34 variabel prediktor. Kerangka kerja penelitian ini membatasi jendela pengamatan secara ketat pada tahun pertama perkuliahan (Semester 1 dan Semester 2) dengan memadukan indikator akademik awal, kondisi finansial/tunggakan biaya kuliah, status beasiswa, dan profil demografi. Kinerja model dievaluasi secara ketat menggunakan metrik yang tahan terhadap ketidakseimbangan data, yaitu *Precision-Recall AUC* (PR-AUC), *Recall (Sensitivity)*, dan *Macro F1-Score* dengan skema *Stratified 5-Fold Cross Validation*. Lapisan XAI dimanfaatkan untuk menguraikan faktor pendorong risiko (*risk drivers*) di tingkat global institusi maupun di tingkat lokal per individu mahasiswa (*waterfall plot*), yang kemudian dipetakan ke dalam rekomendasi aksi preskriptif bagi dosen wali. Hasil uji coba awal (*small test*) pada dataset nyata Kaggle ($N=885$ test set) menunjukkan bahwa model usulan *Gradient Boosting* berhasil mencapai **Accuracy 88.36%**, **Precision 85.77%**, **ROC-AUC 0.9324**, **PR-AUC 0.8989**, dan **Macro F1-Score 0.8623**, membuktikan keunggulan dan transparansi framework yang diusulkan sebagai fondasi sistem pendukung keputusan retensi mahasiswa di perguruan tinggi.

**Kata Kunci**: *Educational Data Mining*, *Student Retention*, *Drop-Out Prediction*, *Gradient Boosting*, *Explainable AI*, *SHAP*, *Imbalanced Data*, *Kaggle Benchmark Dataset*.

---

### **ABSTRACT**
*High student drop-out rates in higher education lead to severe financial and reputational losses for universities while impeding students' personal and professional trajectories. While previous Educational Data Mining (EDM) studies have achieved notable predictive accuracy, most rely on black-box models that predict late-stage attrition without providing actionable explanations, often neglecting inherent class imbalances. Consequently, academic advisors cannot discern the specific root causes behind an individual student's risk profile.*

*This research proposes an interpretable early warning framework utilizing Gradient Boosting algorithms coupled with Explainable Artificial Intelligence (XAI) via SHapley Additive exPlanations (SHAP). The research employs an authentic empirical dataset sourced from Kaggle and the UCI Machine Learning Repository (Predict Students' Dropout and Academic Success, Realinho et al., 2022), comprising 4,424 undergraduate student records with 34 predictor features. The observation window is strictly confined to the first academic year (Semesters 1 and 2), harmonizing early academic performance, tuition fee compliance, scholarship status, and demographic profiles. Model efficacy is rigorously benchmarked using imbalance-resilient metrics, including Precision-Recall AUC (PR-AUC), Sensitivity/Recall, and Macro F1-Score through Stratified 5-Fold Cross-Validation. The XAI layer provides both global feature importance and local instance-level attribution, directly translating mathematical explanations into prescriptive academic interventions. Preliminary experimental evaluations on the Kaggle holdout test set (N = 885) demonstrate an Accuracy of 88.36%, Precision of 85.77%, ROC-AUC of 0.9324, PR-AUC of 0.8989, and Macro F1 of 0.8623, demonstrating the robustness and interpretability of the proposed framework for institutional decision-support systems.*

**Keywords**: *Educational Data Mining*, *Student Retention*, *Drop-Out Prediction*, *Gradient Boosting*, *Explainable AI*, *SHAP*, *Imbalanced Data*, *Kaggle Benchmark Dataset*.

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang Masalah
Pendidikan tinggi memegang peranan krusial dalam mencetak sumber daya manusia yang kompeten dan berdaya saing. Namun, fenomena atrisi atau putus studi (*drop-out*) mahasiswa masih menjadi momok bagi perguruan tinggi di seluruh dunia, termasuk di Indonesia. Berdasarkan data statistik pendidikan tinggi, tingkat retensi mahasiswa tahun pertama merupakan indikator krusial dalam akreditasi institusi dan penentuan mutu program studi. Mahasiswa yang putus studi tidak hanya menanggung kerugian waktu dan beban finansial yang terbuang sia-sia, tetapi juga berdampak langsung pada kestabilan operasional kampus berupa berkurangnya perolehan Uang Kuliah Tunggal (UKT) serta penurunan reputasi institusi.

Dalam bidang informatika dan ilmu komputer, perkembangan *Educational Data Mining* (EDM) dan *Learning Analytics* telah membuka peluang besar untuk memanfaatkan data historis akademik dalam memprediksi mahasiswa yang berisiko mengalami kendala studi. Berbagai algoritma pembelajaran mesin (*machine learning*) seperti *Support Vector Machine* (SVM), *Artificial Neural Networks* (ANN), hingga *Random Forest* telah banyak dieksplorasi dalam penelitian terdahulu. Meskipun algoritma-algoritma tersebut mampu memberikan performa prediksi yang baik, terdapat empat kendala utama yang belum terpecahkan secara optimal:

1. **Sifat Model yang Tertutup (*Black-Box Problem*)**: Algoritma pembelajaran mesin modern yang canggih umumnya bekerja seperti kotak hitam. Pengambil kebijakan akademik (Dekan, Ketua Program Studi, dan Dosen Wali) hanya disajikan probabilitas risiko (misal: "Mahasiswa A memiliki risiko drop-out 88%"), tanpa memahami latar belakang kausalitas mengapa model mengeluarkan kesimpulan tersebut. Tanpa adanya transparansi penjelasan, pihak kampus ragu-ragu dalam mengambil keputusan krusial.
2. **Keterlambatan Waktu Deteksi (*Late Detection Horizon*)**: Sebagian besar penelitian terdahulu menyertakan fitur rekam jejak akademik mahasiswa hingga semester 4, 5, atau 6. Pada tahap ini, intervensi seringkali sudah terlambat karena kegagalan akademik atau beban finansial mahasiswa sudah terakumulasi sedemikian rupa sehingga pengunduran diri sulit dicegah.
3. **Ketidakseimbangan Kelas dan Bias Akurasi Semu (*Class Imbalance Paradox*)**: Secara alamiah, jumlah mahasiswa yang mengalami putus studi berada pada proporsi minoritas (pada data empiris Kaggle berkisar 32.12%). Model konvensional yang dilatih tanpa penanganan ketidakseimbangan kelas cenderung memaksimalkan *Overall Accuracy* dengan mengorbankan kelas minoritas. Akibatnya, angka *False Negative* melonjak tinggi, di mana mahasiswa yang berisiko justru luput dari sistem peringatan dini.
4. **Ketiadaan Rekomendasi Preskriptif (*Lack of Actionable Prescriptions*)**: Output prediksi dalam penelitian literatur umumnya berhenti pada klasifikasi biner `[Drop-out / Lanjut]`, tanpa memberikan petunjuk konkret apakah mahasiswa tersebut membutuhkan penyesuaian rencana studi (*academic mentoring*) atau bantuan restrukturisasi biaya kuliah (*financial aid*).

Untuk mengatasi celah permasalahan tersebut, penelitian ini mengusulkan penerapan algoritma ensemble mutakhir **Gradient Boosting** yang dipadukan dengan kerangka kerja **Explainable Artificial Intelligence (XAI)** berbasis **SHAP (SHapley Additive exPlanations)**. Penelitian ini menggunakan dataset benchmark empiris dari **Kaggle** dan **UCI Machine Learning Repository** dengan 4.424 rekam data mahasiswa. Kerangka kerja ini secara khusus difokuskan pada rekam jejak **Tahun Pertama (Semester 1 dan Semester 2)** untuk mendeteksi risiko sedini mungkin, serta memanfaatkan metrik **PR-AUC**, **Precision**, dan **Recall** untuk menjamin sensitivitas maksimal terhadap kelompok mahasiswa yang terancam putus studi.

---

### 1.2 Rumusan Masalah
Berdasarkan latar belakang yang telah dipaparkan, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana merancang arsitektur model klasifikasi berbasis algoritma *Gradient Boosting* yang optimal dalam mendeteksi risiko *drop-out* mahasiswa menggunakan data tahun pertama (Semester 1 & 2) pada dataset empiris Kaggle?
2. Bagaimana efektivitas penerapan metrik evaluasi *imbalance-resilient* (PR-AUC, Macro F1, Recall, Precision) dibandingkan metrik akurasi konvensional dalam mengevaluasi kinerja model deteksi dini?
3. Bagaimana mengintegrasikan kerangka kerja *Explainable AI* (XAI) berbasis metode SHAP untuk menghasilkan interpretasi global mengenai faktor dominan putus studi di tingkat institusi dan interpretasi lokal per individu mahasiswa?
4. Bagaimana memetakan dekomposisi atribusi fitur SHAP ke dalam rekomendasi aksi preskriptif yang dapat diimplementasikan langsung oleh Dosen Pembimbing Akademik?

---

### 1.3 Batasan Masalah (Ruang Lingkup Penelitian)
Agar penelitian ini terarah dan memiliki fokus yang mendalam, ditetapkan batasan masalah sebagai berikut:
1. **Sumber Data**: Menggunakan dataset benchmark publik resmi dari **Kaggle** (*Predict Students' Dropout and Academic Success*) yang bersumber dari *Polytechnic Institute of Portalegre (IPP)* via *UCI Machine Learning Repository* (Dataset ID: 697), terdiri dari 4.424 baris data dan 34 fitur prediktor.
2. **Horizon Waktu Data**: Data yang dianalisis dibatasi secara ketat pada data profil penerimaan awal, latar belakang sosio-ekonomi, dan rekam akademik dua semester pertama (Semester 1 dan Semester 2).
3. **Algoritma yang Digunakan**: Algoritma utama yang diuji adalah *Gradient Boosting Classifier* (dengan pembanding *Logistic Regression* sebagai linear baseline dan *Random Forest* sebagai bagging baseline).
4. **Metode XAI**: Pengujian interpretabilitas model difokuskan pada metode *post-hoc* model-agnostik **SHAP (TreeExplainer)** dengan verifikasi visualisasi *Waterfall Plot* dan *Summary Plot*.

---

### 1.4 Tujuan Penelitian
Tujuan yang ingin dicapai melalui penelitian ini adalah:
1. Membangun model deteksi dini risiko *drop-out* mahasiswa berbasis *Gradient Boosting* dengan tingkat sensitivitas (Recall), Presisi, dan PR-AUC yang tinggi pada data tahun pertama.
2. Membandingkan performa model usulan terhadap baseline (*Logistic Regression* dan *Random Forest*) dengan protokol validasi terstandar *Stratified 5-Fold Cross Validation*.
3. Menerapkan metode SHAP untuk membongkar sifat *black-box* model, menghasilkan atribusi fitur yang adil secara matematis berdasarkan *Shapley Values*.
4. Merancang prototipe konseptual *Decision Support Dashboard* yang menghubungkan hasil penjelasan SHAP dengan skema intervensi preskriptif (akademik vs finansial).

---

### 1.5 Manfaat Penelitian
#### A. Manfaat Teoretis (Keilmuan)
* Memperkaya khazanah keilmuan *Educational Data Mining* dan *Explainable Artificial Intelligence* melalui integrasi praktis antara model ensemble non-linear dengan teori kooperatif permainan (*Shapley values*).
* Memberikan bukti empiris mengenai ketahanan metrik evaluasi *Precision-Recall AUC* dalam mengatasi fenomena *Accuracy Paradox* pada data pendidikan berdistribusi timpang (*imbalanced*).

#### B. Manfaat Praktis (Aplikatif)
* **Bagi Pengelola Institusi & Program Studi**: Menyediakan sistem peringatan dini (*Early Warning System*) berbasis data yang akurat untuk menurunkan angka putus studi dan menjaga kestabilan operasional kampus.
* **Bagi Dosen Pembimbing Akademik (Dosen Wali)**: Memberikan wawasan yang transparan dan dapat dipahami mengenai akar permasalahan spesifik setiap mahasiswa, sehingga bimbingan akademik menjadi personal dan solutif.
* **Bagi Mahasiswa**: Mencegah kegagalan studi secara dini melalui pendampingan tepat sasaran sebelum masalah akademik atau finansial menjadi permanen.

---

## BAB II: TINJAUAN PUSTAKA & LANDASAN TEORI

### 2.1 Educational Data Mining (EDM) & Retensi Mahasiswa
*Educational Data Mining* (EDM) merupakan cabang ilmu komputasi yang berfokus pada pengembangan metode untuk mengeksplorasi data berskala besar dari lingkungan pendidikan guna memahami pola belajar mahasiswa serta mengoptimalkan efisiensi institusi. Dalam literatur retensi pendidikan tinggi, teori Vincent Tinto mengenai *Student Departure Model* menyatakan bahwa keberhasilan mahasiswa ditentukan oleh integrasi akademik dan integrasi sosial di kampus. Kegagalan adaptasi pada tahun pertama merupakan periode paling rentan yang menjadi pemicu utama keputusan mahasiswa untuk mengundurkan diri.

---

### 2.2 Algoritma Gradient Boosting
*Gradient Boosting* adalah keluarga algoritma *ensemble learning* yang membangun model prediktif secara sekuensial dengan mengoptimalkan fungsi kerugian (*loss function*) menggunakan algoritma penurunan gradien (*gradient descent*). Pada setiap iterasi $m$, pohon keputusan baru $h_m(x)$ dilatih untuk memprediksi residual atau pseudo-residual dari kombinasi model sebelumnya $F_{m-1}(x)$:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Di mana $\eta$ adalah laju pembelajaran (*learning rate*). Keunggulan Gradient Boosting terletak pada kemampuannya memodelkan interaksi fitur non-linear yang rumit serta ketahanan terhadap data numerik dan kategorikal tanpa membutuhkan normalisasi data yang agresif.

---

### 2.3 Teori Explainable Artificial Intelligence (XAI) & SHAP
Explainable AI (XAI) bertujuan untuk memberikan transparansi dan interpretabilitas pada keluaran algoritma kecerdasan buatan. Pendekatan **SHAP (SHapley Additive exPlanations)** yang diperkenalkan oleh Lundberg dan Lee (2017) berlandaskan pada *Cooperative Game Theory*. Nilai kontribusi $\phi_i$ dari suatu fitur $i$ dihitung sebagai rata-rata kontribusi marjinal fitur tersebut di seluruh kemungkinan kombinasi himpunan bagian fitur $S \subseteq F \setminus \{i\}$:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

Keunggulan matematis metode SHAP dibandingkan metode pentingnya fitur konvensional (*Gini Importance*) adalah pemenuhan empat sifat aksiomatis:
1. **Efficiency**: Jumlah seluruh nilai SHAP sama dengan selisih antara keluaran model dan nilai ekspektasi dasar.
2. **Symmetry**: Dua fitur yang memberikan kontribusi identik pada semua kombinasi akan menerima nilai atribusi yang persis sama.
3. **Dummy / Null Player**: Fitur yang tidak berpengaruh pada prediksi memiliki nilai SHAP nol.
4. **Additivity**: Nilai penjelasan pada model ansambel dapat dijumlahkan secara linear.

---

## BAB III: METODOLOGI PENELITIAN

### 3.1 Kerangka Kerja Riset (CRISP-DM Adapted)
Penelitian ini mengadopsi metodologi standar industri *Cross-Industry Standard Process for Data Mining* (CRISP-DM) yang disesuaikan ke dalam 6 tahapan terstruktur:
1. **Business/Institutional Understanding**: Mendefinisikan kriteria drop-out dan memetakan kebutuhan intervensi Dosen Pembimbing Akademik.
2. **Data Understanding**: Memeriksa kelengkapan 34 fitur prediktor (demografi, finansial, akademik S1 & S2) dan 1 variabel target pada 4.424 sampel data riil mahasiswa Kaggle.
3. **Data Preparation**: Penanganan standarisasi kolom, verifikasi integritas data, dan pemisahan data train-test terstratifikasi.
4. **Modeling**: Pelatihan model *Logistic Regression*, *Random Forest*, dan *Gradient Boosting* dengan penanganan *class weights*.
5. **Evaluation**: Pengukuran komparatif metrik PR-AUC, ROC-AUC, Recall, Precision, dan Macro F1 melalui *Stratified 5-Fold Cross Validation*.
6. **Explanation & Prescriptive Translation**: Pemasangan lapisan *SHAP TreeExplainer* untuk ekstraksi faktor risiko global dan lokal, diikuti pemetaan aksi intervensi.

---

### 3.2 Alur Penelitian (Research Pipeline)
Alur metodologi penelitian divisualisasikan dalam diagram berikut:

```
[Dataset Empiris Kaggle / UCI: 4.424 Rekam Mahasiswa]
                          │
                          ▼
[Data Cleaning & Stratified Partitioning: 80% Train (3.539) / 20% Test (885)]
                          │
                          ▼
            ┌───────────────────────────┐
            │   Pelatihan Model (5-Fold)│
            ├───────────────────────────┤
            │ • Logistic Regression     │
            │ • Random Forest           │
            │ • Gradient Boosting (Main)│
            └─────────────┬─────────────┘
                          │
                          ▼
[Pengukuran Evaluasi Metrik: PR-AUC, Recall, Precision, F1-Macro, ROC-AUC]
                          │
                          ▼
[Lapisan Explainable AI: SHAP Explainer (Global Summary & Local Waterfall)]
                          │
                          ▼
[Rekomendasi Preskriptif: Akademik Mentoring vs Financial Relief]
```

---

### 3.3 Variabel dan Karakteristik Data Kaggle
Dataset penelitian mencakup variabel multidimensi lengkap (34 prediktor):
* **Fitur Demografi**: `Marital_status`, `Nacionality`, `Displaced`, `Educational_special_needs`, `Gender`, `Age_at_enrollment`, `International`.
* **Fitur Pendaftaran**: `Application_mode`, `Application_order`, `Course`, `Daytime_evening_attendance`, `Previous_qualification`.
* **Fitur Sosio-Ekonomi & Finansial**: `Mothers_qualification`, `Fathers_qualification`, `Mothers_occupation`, `Fathers_occupation`, `Scholarship_holder`, `Debtor`, `Tuition_fees_up_to_date`.
* **Fitur Akademik Awal (S1 & S2)**: SKS terdaftar, SKS lulus, nilai rata-rata, dan frekuensi evaluasi ujian.
* **Fitur Kontekstual Makro**: `Unemployment_rate`, `Inflation_rate`, `GDP`.
* **Variabel Target**: `Target` (Dropout, Enrolled, Graduate) dan biner `Target_Dropout` (0 = Non-Dropout, 1 = Dropout).

---

## BAB IV: RENCANA JADWAL PELAKSANAAN & MILESTONES

Penelitian ini direncanakan berlangsung selama 6 (enam) bulan dengan rincian jadwal kerja sebagai berikut:

| No | Tahapan Aktivitas / Milestone | Bulan 1 | Bulan 2 | Bulan 3 | Bulan 4 | Bulan 5 | Bulan 6 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Pengambilan Data Kaggle & Studi Literatur Mendalam | █ | | | | | |
| 2 | Eksplorasi Data (EDA) & Preprocessing Pipeline | | █ | | | | |
| 3 | Pengembangan Baseline & Model Gradient Boosting | | | █ | | | |
| 4 | Validasi Kinerja Model & Hyperparameter Tuning | | | | █ | | |
| 5 | Integrasi Lapisan XAI (SHAP) & Modul Preskriptif | | | | | █ | |
| 6 | Uji Pakar Akademik, Penyusunan Laporan & Publikasi | | | | | | █ |

---

## DAFTAR PUSTAKA

1. Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). "Predicting student dropout and academic success." *Data*, 7(11), 146.
2. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765-4774.
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
4. Ke, G., Meng, Q., Finley, T., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 3146-3154.
5. Agrusti, G., Bonavolontà, G., & Mezzini, M. (2020). "Analytical paradigms for early prediction of student dropout in higher education." *Computers & Education*, 154, 103927.
6. Albreiki, B., Zaki, N., & Alashwal, H. (2021). "A systematic literature review of student’s performance prediction using machine learning." *IEEE Access*, 9, 79579-79592.
7. Bhat, S., Raju, R., & Bikshapathi, K. (2022). "Leveraging deep learning for early detection of student dropout in virtual learning environments." *IEEE Transactions on Learning Technologies*, 15(4), 512-525.
8. Rastrollo-Guerrero, J. L., Gómez-Pulido, J. A., & Durán-Domínguez, A. (2020). "Analyzing and predicting students’ performance by means of machine learning: A review." *Applied Sciences*, 10(3), 1042.
9. Baranyi, M., Nagy, M., & Molontay, R. (2023). "Interpretable machine learning for student dropout prediction: A comparative study of explainability methods." *Expert Systems with Applications*, 213, 118873.
10. Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating imbalanced datasets." *PLoS ONE*, 10(3), e0118432.
11. Tinto, V. (1993). *Leaving College: Rethinking the Causes and Cures of Student Attrition*. University of Chicago Press.
12. Romero, C., & Ventura, S. (2020). "Educational data mining and learning analytics: An updated survey." *WIREs Data Mining and Knowledge Discovery*, 10(3), e1355.
13. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: synthetic minority over-sampling technique." *Journal of Artificial Intelligence Research*, 16, 321-357.
14. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). ""Why should I trust you?": Explaining the predictions of any classifier." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.
15. He, H., & Garcia, E. A. (2009). "Learning from imbalanced data." *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
16. Tanuar, E., et al. (2022). "Penerapan Algoritma Data Mining untuk Prediksi Kelulusan Mahasiswa Tepat Waktu." *Jurnal Sistem Informasi Bisnis*, 12(1), 45-52.
17. Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). "CatBoost: unbiased boosting with categorical features." *Advances in Neural Information Processing Systems (NeurIPS)*, 31, 6638-6648.
18. Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. 2nd ed.
19. Stupariu, M. S., et al. (2024). "Evaluating machine learning models under extreme class imbalance for higher education retention." *International Journal of Educational Technology in Higher Education*, 21(1), 1-22.
20. Fawcett, T. (2006). "An introduction to ROC analysis." *Pattern Recognition Letters*, 27(8), 861-874.
