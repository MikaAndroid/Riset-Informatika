"""
Script: generate_dataset.py
Deskripsi: Menghasilkan dataset terstruktur untuk riset deteksi dini risiko drop-out mahasiswa
berdasarkan benchmark UCI 'Predict Students Dropout and Academic Success' (Realinho et al., 2022).
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Identitas & Demografi
    student_ids = [f"STD_{1000 + i}" for i in range(n_samples)]
    gender = np.random.choice([0, 1], size=n_samples, p=[0.48, 0.52]) # 0: Female, 1: Male
    age_at_enrollment = np.clip(np.random.exponential(scale=3.5, size=n_samples) + 18, 17, 45).astype(int)
    marital_status = np.random.choice([1, 2, 4], size=n_samples, p=[0.86, 0.10, 0.04]) # 1: Single, 2: Married, 4: Divorced
    displaced = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55]) # 1: Merantau / Luar Daerah
    educational_special_needs = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    
    # 2. Latar Belakang Sosio-Ekonomi & Finansial
    scholarship_holder = np.random.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    
    # Debtor status berkolerasi dengan beasiswa dan usia
    debtor_prob = np.where(scholarship_holder == 1, 0.05, 0.22)
    debtor_prob = np.clip(debtor_prob + (age_at_enrollment > 24) * 0.12, 0.02, 0.60)
    debtor = np.random.binomial(1, debtor_prob)
    
    # Keteraturan UKT / SPP (Tuition fees up to date)
    tuition_up_to_date_prob = np.where(debtor == 1, 0.18, 0.92)
    tuition_fees_up_to_date = np.random.binomial(1, tuition_up_to_date_prob)
    
    parents_higher_edu = np.random.choice([0, 1], size=n_samples, p=[0.62, 0.38])
    
    # 3. Kualifikasi Masuk & Akademik Awal
    admission_grade = np.clip(np.random.normal(loc=126, scale=17, size=n_samples), 95, 195).round(1)
    
    # 4. Kinerja Akademik Semester 1 (Early Indicators)
    s1_enrolled = np.random.choice([5, 6, 7], size=n_samples, p=[0.15, 0.75, 0.10])
    
    # Academic capacity score tersembunyi untuk korelasi realistis
    academic_latent = (
        (admission_grade - 125) / 22.0 
        - 1.1 * debtor 
        + 0.6 * scholarship_holder 
        - 0.5 * (age_at_enrollment > 24) 
        + np.random.normal(0, 0.9, size=n_samples)
    )
    
    # Approval rate Semester 1
    s1_pass_ratio = 1.0 / (1.0 + np.exp(-(academic_latent + 0.3)))
    s1_approved = np.clip(np.round(s1_enrolled * s1_pass_ratio), 0, s1_enrolled).astype(int)
    
    s1_grade_base = 10.0 + academic_latent * 2.1 + (s1_approved / s1_enrolled) * 3.8
    s1_grade = np.where(s1_approved == 0, 0.0, np.clip(s1_grade_base + np.random.normal(0, 1.1, size=n_samples), 10.0, 19.5)).round(2)
    s1_evaluations = np.clip(s1_enrolled + np.random.poisson(lam=1.8, size=n_samples), s1_enrolled, 14)
    
    # 5. Kinerja Akademik Semester 2
    s2_enrolled = np.where(s1_approved <= 2, np.random.choice([4, 5, 6], size=n_samples, p=[0.4, 0.4, 0.2]), s1_enrolled)
    
    academic_latent_s2 = academic_latent + 0.5 * ((s1_approved / s1_enrolled) - 0.6) + np.random.normal(0, 0.7, size=n_samples)
    s2_pass_ratio = 1.0 / (1.0 + np.exp(-(academic_latent_s2 + 0.2)))
    s2_approved = np.clip(np.round(s2_enrolled * s2_pass_ratio), 0, s2_enrolled).astype(int)
    
    s2_grade_base = 10.0 + academic_latent_s2 * 2.1 + (s2_approved / s2_enrolled) * 3.8
    s2_grade = np.where(s2_approved == 0, 0.0, np.clip(s2_grade_base + np.random.normal(0, 1.1, size=n_samples), 10.0, 19.5)).round(2)
    s2_evaluations = np.clip(s2_enrolled + np.random.poisson(lam=1.8, size=n_samples), s2_enrolled, 14)
    
    # 6. Fitur Ekonomi Makro (Konteks Kontekstual Institusi)
    unemployment_rate = np.random.choice([8.9, 10.8, 12.4, 13.9, 15.5], size=n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    inflation_rate = np.random.choice([-0.8, 0.5, 1.4, 2.8, 3.7], size=n_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1])
    gdp = np.random.choice([-1.7, 0.3, 1.8, 2.0, 3.5], size=n_samples, p=[0.15, 0.25, 0.35, 0.15, 0.1])
    
    # 7. Penentuan Target Riset (Dropout vs Graduate vs Enrolled)
    # Log-odds risiko drop-out
    dropout_score = (
        - 0.50 
        + 2.60 * (1 - tuition_fees_up_to_date)
        + 1.85 * debtor
        - 1.30 * scholarship_holder
        - 0.50 * s1_approved
        - 0.65 * s2_approved
        - 0.14 * (s1_grade - 10.0)
        - 0.16 * (s2_grade - 10.0)
        + 0.08 * (age_at_enrollment - 20)
        + 0.25 * gender
        - 0.40 * parents_higher_edu
        + np.random.normal(0, 0.9, size=n_samples)
    )
    
    p_dropout = 1.0 / (1.0 + np.exp(-dropout_score))
    is_dropout = (p_dropout > 0.45).astype(int)
    
    # Kategori Target (3 kelas & binary)
    target_class = []
    for d, p, s1_app, s2_app in zip(is_dropout, p_dropout, s1_approved, s2_approved):
        if d == 1:
            target_class.append("Dropout")
        elif p < 0.25 and (s1_app + s2_app >= 9):
            target_class.append("Graduate")
        else:
            target_class.append("Enrolled" if np.random.rand() < 0.40 else "Graduate")
            
    # Buat DataFrame
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": gender, # 1: Male, 0: Female
        "Age_at_Enrollment": age_at_enrollment,
        "Marital_Status": marital_status,
        "Displaced": displaced,
        "Special_Needs": educational_special_needs,
        "Parents_Higher_Education": parents_higher_edu,
        "Scholarship_Holder": scholarship_holder,
        "Debtor": debtor,
        "Tuition_Fees_Up_To_Date": tuition_fees_up_to_date,
        "Admission_Grade": admission_grade,
        "Curricular_Units_1st_Sem_Enrolled": s1_enrolled,
        "Curricular_Units_1st_Sem_Evaluations": s1_evaluations,
        "Curricular_Units_1st_Sem_Approved": s1_approved,
        "Curricular_Units_1st_Sem_Grade": s1_grade,
        "Curricular_Units_2nd_Sem_Enrolled": s2_enrolled,
        "Curricular_Units_2nd_Sem_Evaluations": s2_evaluations,
        "Curricular_Units_2nd_Sem_Approved": s2_approved,
        "Curricular_Units_2nd_Sem_Grade": s2_grade,
        "Unemployment_Rate": unemployment_rate,
        "Inflation_Rate": inflation_rate,
        "GDP": gdp,
        "Target": target_class,
        "Target_Dropout": is_dropout # 1: Dropout, 0: Non-Dropout (Graduate/Enrolled)
    })
    
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    df = generate_student_dataset(n_samples=1500, random_seed=2026)
    csv_path = os.path.join(output_dir, "student_dropout_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"[OK] Dataset berhasil digenerate di: {csv_path}")
    print(f"Total baris: {len(df)}, Kolom: {df.shape[1]}")
    print("\nDistribusi Target 3-Kelas:")
    print(df["Target"].value_counts(normalize=True).round(3) * 100)
    print("\nDistribusi Target Biner (Target_Dropout):")
    print(df["Target_Dropout"].value_counts(normalize=True).round(3) * 100)
