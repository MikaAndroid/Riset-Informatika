# 📚 02 - Dataset Rencana Riset Kecil
## Dataset Empiris: Predict Students' Dropout and Academic Success

Folder ini berisi berkas dataset nyata yang digunakan dalam rencana riset, kamus data lengkap (*data dictionary*), statistik deskriptif, dan script pengunduh data otomatis.

---

### 📂 Berkas di Folder Ini
* **`student_dropout_dataset.csv`**: File dataset riil berformat CSV (4.424 baris data, 36 kolom).
* **`data_dictionary.md`**: Kamus data terperinci mendokumentasikan 34 fitur prediktor dan variabel target.
* **`descriptive_statistics.csv`**: Rangkuman statistik deskriptif numerik (mean, std, min, median, max).
* **`fetch_kaggle_dataset.py`**: Script Python untuk mengunduh dataset dari repositori resmi Kaggle/UCI.

---

### 🌐 Sumber Resmi Dataset
* **Kaggle**: [Predict Students' Dropout and Academic Success (thedevastator)](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
* **Kaggle Competition**: [Playground Series s4e6](https://www.kaggle.com/competitions/playground-series-s4e6)
* **UCI Repository**: [UCI Machine Learning Repository - Dataset ID: 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
* **Institusi Asal**: *Polytechnic Institute of Portalegre (IPP)*, Portugal.

---

### 📊 Ringkasan Karakteristik Data
* **Jumlah Observasi**: 4.424 Mahasiswa
* **Jumlah Atribut**: 36 Kolom (34 Fitur Prediktor, 1 Target Asli 3-Kelas, 1 Target Biner Deteksi Dini)
* **Distribusi Target Asli**:
  * `Graduate`: 2.209 mahasiswa (49.93%)
  * `Dropout`: 1.421 mahasiswa (32.12%)
  * `Enrolled`: 794 mahasiswa (17.95%)
* **Distribusi Target Biner (`Target_Dropout`)**:
  * `Non-Dropout (0)`: 3.003 mahasiswa (67.88%)
  * `Dropout (1)`: 1.421 mahasiswa (32.12%)
  * Imbalance Ratio: $\approx 1 : 2.11$

---

### 🔄 Cara Mengunduh Ulang Dataset
Jalankan script berikut:
```bash
python fetch_kaggle_dataset.py
```
