# 📚 Kamus Data (Data Dictionary)
## Riset: Deteksi Dini Risiko Mahasiswa Drop-Out Menggunakan Gradient Boosting & Explainable AI (XAI) Berbasis SHAP

Dokumen ini mendokumentasikan dataset empiris **asli (bukan sintetis)** yang bersumber dari repositori publik bereputasi internasional: **Kaggle** dan **UCI Machine Learning Repository**.

---

### 1. Sumber dan Metadata Resmi Dataset

| Parameter | Keterangan Sumber Resmi |
| :--- | :--- |
| **Nama Dataset di Kaggle** | [*Predict Students' Dropout and Academic Success*](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) |
| **Kompetisi Resmi Kaggle** | [*Kaggle Playground Series s4e6: Predicting Student Dropout and Academic Success*](https://www.kaggle.com/competitions/playground-series-s4e6) |
| **Repositori Resmi UCI ML** | [*UCI Machine Learning Repository - Dataset ID 697*](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) |
| **Publikasi Sumber Asli** | Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predicting Student Dropout and Academic Success*. **Data**, 7(11), 146. MDPI. |
| **Institusi Asal Data** | *Polytechnic Institute of Portalegre (IPP)*, Portugal |
| **Nama Berkas Lokal** | `data/raw/student_dropout_dataset.csv` |
| **Total Observasi Riil** | **4.424 Mahasiswa** |
| **Total Variabel** | **36 Kolom** (34 Fitur Prediktor, 1 Target Multikelas, 1 Target Biner) |
| **Domain Riset** | *Educational Data Mining* (EDM) & *Learning Analytics* |

---

### 2. Distribusi Variabel Sasaran (Target)

#### A. Target Asli 3-Kelas (`Target`)
| Kelas | Definisi Kondisi Akhir Mahasiswa | Jumlah Mahasiswa | Persentase Riil |
| :--- | :--- | :---: | :---: |
| **Graduate** | Berhasil menyelesaikan studi dan diwisuda (*Graduated*) | 2.209 | 49.93% |
| **Dropout** | Putus studi / mengundurkan diri sebelum lulus (*Dropped Out*) | 1.421 | 32.12% |
| **Enrolled** | Masih aktif terdaftar dan menempuh perkuliahan (*Ongoing*) | 794 | 17.95% |
| **Total** | | **4.424** | **100.00%** |

#### B. Target Biner Deteksi Dini (`Target_Dropout`)
Dalam penelitian sistem peringatan dini (*Early Warning System*), fokus deteksi diarahkan secara biner untuk mengidentifikasi kelompok rentan yang memerlukan intervensi segera:
* **Kelas 1 (*Dropout*)**: 1.421 mahasiswa (**32.12%**) $\rightarrow$ Kelompok kritis yang membutuhkan intervensi akademik/finansial preventif.
* **Kelas 0 (*Non-Dropout: Graduate / Enrolled*)**: 3.003 mahasiswa (**67.88%**) $\rightarrow$ Mahasiswa yang bertahan atau berhasil lulus.
* **Rasio Ketidakseimbangan (*Imbalance Ratio*)**: $\approx 1 : 2.11$.

---

### 3. Taksonomi dan Deskripsi Lengkap 34 Fitur Prediktor

#### A. Latar Belakang Demografi & Personal Mahasiswa
| No | Nama Kolom | Tipe Data | Skala | Rentang / Nilai | Deskripsi & Definisi Operasional |
|:---|:---|:---:|:---:|:---:|:---|
| 1 | `Marital_status` | Integer | Nominal | 1 - 6 | Status perkawinan (1: Belum menikah, 2: Menikah, 3: Duda/Janda, 4: Cerai, 5: Tinggal bersama, 6: Pisah resmi). |
| 2 | `Nacionality` | Integer | Nominal | 1 - 21 | Kewarganegaraan mahasiswa (1: Domestik/Portugal, lainnya: Internasional/Komunitas Lusophone). |
| 3 | `Displaced` | Integer | Biner | 0, 1 | Status perantauan (1: Merantau/luar kota, 0: Domisili lokal/bersama keluarga). |
| 4 | `Educational_special_needs` | Integer | Biner | 0, 1 | Kebutuhan pendidikan khusus / penyandang disabilitas (1: Ya, 0: Tidak). |
| 5 | `Gender` | Integer | Biner | 0, 1 | Jenis kelamin (1: Laki-laki, 0: Perempuan). |
| 6 | `Age_at_enrollment` | Integer | Rasio | 17 - 70 | Usia mahasiswa saat registrasi pendaftaran pertama kali. |
| 7 | `International` | Integer | Biner | 0, 1 | Status mahasiswa internasional (1: Ya, 0: Tidak). |

#### B. Data Pendaftaran & Kualifikasi Masuk (*Admission Data*)
| No | Nama Kolom | Tipe Data | Skala | Rentang / Nilai | Deskripsi & Definisi Operasional |
|:---|:---|:---:|:---:|:---:|:---|
| 8 | `Application_mode` | Integer | Nominal | 1 - 18 | Jalur seleksi pendaftaran masuk (Jalur reguler fase 1-3, alih jenjang, diploma, dll). |
| 9 | `Application_order` | Integer | Ordinal | 0 - 9 | Urutan prioritas pilihan program studi oleh mahasiswa (0: Pilihan pertama, 9: Pilihan terakhir). |
| 10 | `Course` | Integer | Nominal | 1 - 17 | Kode program studi yang ditempuh (Teknologi Informasi, Manajemen, Desain, Keperawatan, dll). |
| 11 | `Daytime_evening_attendance` | Integer | Biner | 0, 1 | Waktu perkuliahan (1: Kelas Pagi/Siang, 0: Kelas Malam/Pekerja). |
| 12 | `Previous_qualification` | Integer | Nominal | 1 - 17 | Jenjang pendidikan sebelumnya (Pendidikan Menengah Atas, Kejuruan, Sarjana Muda, dll). |

#### C. Latar Belakang Sosio-Ekonomi & Finansial
| No | Nama Kolom | Tipe Data | Skala | Rentang / Nilai | Deskripsi & Definisi Operasional |
|:---|:---|:---:|:---:|:---:|:---|
| 13 | `Mothers_qualification` | Integer | Nominal | 1 - 29 | Tingkat pendidikan formal ibu kandung. |
| 14 | `Fathers_qualification` | Integer | Nominal | 1 - 29 | Tingkat pendidikan formal ayah kandung. |
| 15 | `Mothers_occupation` | Integer | Nominal | 1 - 32 | Kelompok profesi / pekerjaan ibu kandung. |
| 16 | `Fathers_occupation` | Integer | Nominal | 1 - 32 | Kelompok profesi / pekerjaan ayah kandung. |
| 17 | `Scholarship_holder` | Integer | Biner | 0, 1 | Penerima beasiswa aktif (1: Ya, 0: Tidak). |
| 18 | `Debtor` | Integer | Biner | 0, 1 | Memiliki riwayat tunggakan kewajiban administrasi (1: Ya, 0: Tidak). |
| 19 | `Tuition_fees_up_to_date` | Integer | Biner | 0, 1 | Pembayaran biaya kuliah/UKT lancar dan lunas tepat waktu (1: Ya, 0: Macet). |

#### D. Rekam Akademik Semester 1 (*Early Academic Indicators*)
| No | Nama Kolom | Tipe Data | Skala | Rentang / Nilai | Deskripsi & Definisi Operasional |
|:---|:---|:---:|:---:|:---:|:---|
| 20 | `Curricular_units_1st_sem_credited` | Integer | Rasio | 0 - 20 | Jumlah SKS/modul yang dikonversi/diakui pada Semester 1. |
| 21 | `Curricular_units_1st_sem_enrolled` | Integer | Rasio | 0 - 26 | Jumlah SKS/modul yang dikontrak/diambil pada Semester 1. |
| 22 | `Curricular_units_1st_sem_evaluations` | Integer | Rasio | 0 - 45 | Frekuensi kegiatan asesmen dan ujian yang dihadiri pada Semester 1. |
| 23 | `Curricular_units_1st_sem_approved` | Integer | Rasio | 0 - 26 | Jumlah SKS/modul yang berhasil dinyatakan lulus pada Semester 1. |
| 24 | `Curricular_units_1st_sem_grade` | Float | Rasio | 0.0 - 18.88 | Nilai rata-rata akademik mahasiswa pada Semester 1 (skala 0 - 20). |
| 25 | `Curricular_units_1st_sem_without_evaluations` | Integer | Rasio | 0 - 12 | Jumlah mata kuliah tanpa evaluasi pada Semester 1. |

#### E. Rekam Akademik Semester 2 (*Early Warning Horizon*)
| No | Nama Kolom | Tipe Data | Skala | Rentang / Nilai | Deskripsi & Definisi Operasional |
|:---|:---|:---:|:---:|:---:|:---|
| 26 | `Curricular_units_2nd_sem_credited` | Integer | Rasio | 0 - 19 | Jumlah SKS/modul yang diakui/transfer pada Semester 2. |
| 27 | `Curricular_units_2nd_sem_enrolled` | Integer | Rasio | 0 - 23 | Jumlah SKS/modul yang dikontrak pada Semester 2. |
| 28 | `Curricular_units_2nd_sem_evaluations` | Integer | Rasio | 0 - 33 | Frekuensi asesmen yang dihadiri pada Semester 2. |
| 29 | `Curricular_units_2nd_sem_approved` | Integer | Rasio | 0 - 20 | Jumlah SKS/modul yang berhasil lulus pada Semester 2. |
| 30 | `Curricular_units_2nd_sem_grade` | Float | Rasio | 0.0 - 18.57 | Nilai rata-rata akademik mahasiswa pada Semester 2 (skala 0 - 20). |
| 31 | `Curricular_units_2nd_sem_without_evaluations` | Integer | Rasio | 0 - 12 | Jumlah mata kuliah tanpa evaluasi pada Semester 2. |

#### F. Indikator Kontekstual Makro-Ekonomi
| No | Nama Kolom | Tipe Data | Skala | Rentang / Nilai | Deskripsi & Definisi Operasional |
|:---|:---|:---:|:---:|:---:|:---|
| 32 | `Unemployment_rate` | Float | Rasio | 7.6% - 16.2% | Tingkat pengangguran regional pada tahun kalender akademik. |
| 33 | `Inflation_rate` | Float | Rasio | -0.8% - 3.7% | Laju inflasi tahunan regional. |
| 34 | `GDP` | Float | Rasio | -4.06% - 3.51% | Persentase pertumbuhan Produk Domestik Bruto (PDB) regional. |
