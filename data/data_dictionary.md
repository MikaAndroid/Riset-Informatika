# 📚 Kamus Data (Data Dictionary)
## Riset: Deteksi Dini Risiko Mahasiswa Drop-Out Menggunakan Gradient Boosting & Explainable AI (XAI)

Dataset ini dirancang khusus untuk memodelkan indikator risiko putus studi (*drop-out*) mahasiswa pada tahap awal perkuliahan (Tahun Pertama: Semester 1 & Semester 2). Struktur dan distribusi data diadopsi dari data benchmark internasional (*Polytechnic Institute of Portalegre / UCI Machine Learning Repository*, Realinho et al., 2022) dan disesuaikan dengan konteks administrasi perguruan tinggi.

---

### 1. Ringkasan Metadata Dataset

| Parameter | Spesifikasi |
| :--- | :--- |
| **Nama Berkas** | `student_dropout_dataset.csv` |
| **Lokasi Direktori** | `data/raw/` |
| **Jumlah Baris (Observasi)** | 1.500 Mahasiswa |
| **Jumlah Kolom (Fitur)** | 24 Variabel (1 ID, 21 Prediktor, 2 Target) |
| **Domain Riset** | *Educational Data Mining* (EDM) & *Learning Analytics* |
| **Fokus Evaluasi** | *Early-warning Prediction* (Data Semester 1 & Semester 2) |
| **Rasio Kelas Biner** | 83.9% Non-Dropout (1.258) : 16.1% Dropout (242) |

---

### 2. Taksonomi dan Deskripsi Variabel

#### A. Identitas Mahasiswa
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Student_ID` | String / Object | Nominal | `STD_1000` - `STD_2499` | Pengenal unik anonim mahasiswa (bukan fitur pemodelan). |

#### B. Profil Demografi & Latar Belakang Personal
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Gender` | Integer | Biner | `0` = Perempuan<br>`1` = Laki-laki | Jenis kelamin biologis mahasiswa. |
| `Age_at_Enrollment` | Integer | Rasio | 17 - 45 tahun | Usia mahasiswa saat pertama kali mendaftar/registrasi kuliah. |
| `Marital_Status` | Integer | Nominal | `1` = Belum Menikah (Single)<br>`2` = Menikah<br>`4` = Cerai/Pisah | Status perkawinan resmi saat masa studi. |
| `Displaced` | Integer | Biner | `0` = Tinggal bersama keluarga<br>`1` = Mahasiswa Rantau / Mandiri | Indikator domisili asal mahasiswa terhadap lokasi kampus. |
| `Special_Needs` | Integer | Biner | `0` = Tidak Ada<br>`1` = Memiliki Kebutuhan Khusus | Indikator kebutuhan dukungan pendidikan khusus / disabilitas. |

#### C. Latar Belakang Sosio-Ekonomi & Finansial
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Parents_Higher_Education` | Integer | Biner | `0` = Orang tua non-Sarjana<br>`1` = Minimal salah satu orang tua Sarjana | Riwayat tingkat pendidikan tinggi orang tua (kapital budaya). |
| `Scholarship_Holder` | Integer | Biner | `0` = Tidak menerima beasiswa<br>`1` = Penerima beasiswa (KIP-K / Institusi) | Status penerimaan beasiswa aktif mahasiswa. |
| `Debtor` | Integer | Biner | `0` = Tidak memiliki tunggakan<br>`1` = Memiliki riwayat tunggakan biaya | Indikator apakah mahasiswa memiliki utang/tunggakan administrasi. |
| `Tuition_Fees_Up_To_Date` | Integer | Biner | `0` = Pembayaran UKT/SPP macet<br>`1` = Pembayaran UKT/SPP lunas tepat waktu | Status pelunasan biaya kuliah berjalan (*financial sustainability*). |

#### D. Kualifikasi Masuk & Rekam Akademik Semester 1
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Admission_Grade` | Float | Rasio | 95.0 - 195.0 (skala 200) | Nilai skor ujian masuk seleksi penerimaan mahasiswa baru. |
| `Curricular_Units_1st_Sem_Enrolled` | Integer | Rasio | 5 - 7 SKS/Mata Kuliah | Jumlah mata kuliah / modul yang diambil pada Semester 1. |
| `Curricular_Units_1st_Sem_Evaluations`| Integer | Rasio | 5 - 14 kegiatan | Frekuensi asesmen/evaluasi akademik yang diikuti di Semester 1. |
| `Curricular_Units_1st_Sem_Approved` | Integer | Rasio | 0 - 7 mata kuliah | Jumlah mata kuliah yang dinyatakan lulus pada Semester 1. |
| `Curricular_Units_1st_Sem_Grade` | Float | Rasio | 0.0 - 20.0 | Rata-rata nilai akhir mahasiswa pada Semester 1. |

#### E. Rekam Akademik Semester 2 (Early Warning Milestone)
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Curricular_Units_2nd_Sem_Enrolled` | Integer | Rasio | 4 - 7 SKS/Mata Kuliah | Jumlah mata kuliah yang dikontrak pada Semester 2. |
| `Curricular_Units_2nd_Sem_Evaluations`| Integer | Rasio | 4 - 14 kegiatan | Frekuensi asesmen/evaluasi akademik yang diikuti di Semester 2. |
| `Curricular_Units_2nd_Sem_Approved` | Integer | Rasio | 0 - 7 mata kuliah | Jumlah mata kuliah yang dinyatakan lulus pada Semester 2. |
| `Curricular_Units_2nd_Sem_Grade` | Float | Rasio | 0.0 - 20.0 | Rata-rata nilai akhir mahasiswa pada Semester 2. |

#### F. Indikator Kontekstual Makro-Ekonomi
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Unemployment_Rate` | Float | Rasio | 8.9% - 15.5% | Tingkat pengangguran regional pada tahun registrasi. |
| `Inflation_Rate` | Float | Rasio | -0.8% - 3.7% | Laju inflasi regional tahunan. |
| `GDP` | Float | Rasio | -1.7% - 3.5% | Pertumbuhan Produk Domestik Bruto regional. |

#### G. Variabel Sasaran (Target)
| Nama Kolom | Tipe Data | Skala | Rentang / Kategori | Keterangan Operasional |
| :--- | :--- | :--- | :--- | :--- |
| `Target` | String | Kategorikal | `Dropout`, `Enrolled`, `Graduate` | Klasifikasi multikelas kondisi akademik akhir mahasiswa. |
| `Target_Dropout` | Integer | Biner | `0` = Persisted (Graduate/Enrolled)<br>`1` = Dropout (Putus Studi) | **Target Utama Pemodelan Deteksi Dini Risiko**. |

---

### 3. Distribusi dan Karakteristik Imbalance
Dalam riset ini, fokus deteksi dini diarahkan pada `Target_Dropout`:
- **Kelas Positif (1 - Dropout)**: 242 siswa (~16.1%) $\rightarrow$ Kasus kritis yang membutuhkan intervensi dini.
- **Kelas Negatif (0 - Non-Dropout)**: 1.258 siswa (~83.9%) $\rightarrow$ Siswa yang melanjutkan studi atau berhasil lulus.

Ketidakseimbangan kelas ini (*imbalanced ratio* ~ 1:5.2) mencerminkan kondisi riil di perguruan tinggi dan menjadi dasar pengujian ketahanan metrik evaluasi seperti **PR-AUC**, **Macro F1-Score**, dan teknik penanganan data imbalanced (SMOTE/Class Weights).
