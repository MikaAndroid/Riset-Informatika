"""
Script: eda_and_baseline.py
Deskripsi: Eksplorasi Data Awal (EDA), Pelatihan Baseline Model (Logistic Regression,
Random Forest, Gradient Boosting), dan Perhitungan Lengkap Metrik Evaluasi.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance

def run_experiment():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    data_path = os.path.join(project_root, "data", "raw", "student_dropout_dataset.csv")
    fig_dir = os.path.join(project_root, "outputs", "figures")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 70)
    print("1. MEMUAT DAN MEMERIKSA DATASET...")
    print("=" * 70)
    df = pd.read_csv(data_path)
    print(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    
    # Fitur dan Target
    features = [c for c in df.columns if c not in ["Student_ID", "Target", "Target_Dropout"]]
    X = df[features]
    y = df["Target_Dropout"]
    
    dropout_count = y.sum()
    persist_count = len(y) - dropout_count
    print(f"\nDistribusi Target_Dropout:")
    print(f" - Non-Dropout (0) : {persist_count} ({persist_count/len(y)*100:.2f}%)")
    print(f" - Dropout     (1) : {dropout_count} ({dropout_count/len(y)*100:.2f}%)")
    print(f" - Imbalance Ratio: 1 : {persist_count/dropout_count:.2f}")

    # Simpan statistik deskriptif
    desc = df.describe().round(2)
    desc.to_csv(os.path.join(processed_dir, "descriptive_statistics.csv"))
    print(f"\n[OK] Statistik deskriptif disimpan di data/processed/descriptive_statistics.csv")
    
    # -------------------------------------------------------------
    # Visualisasi 1: Distribusi Kelas & Korelasi Utama
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    counts = df["Target"].value_counts()
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    plt.bar(counts.index, counts.values, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Distribusi Status Kelulusan (3 Kelas)", fontsize=11, fontweight="bold")
    plt.ylabel("Jumlah Mahasiswa")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 15, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    bin_counts = y.value_counts()
    plt.bar(["Non-Dropout (0)", "Dropout (1)"], bin_counts.values, color=["#1b9e77", "#e7298a"], edgecolor="black", alpha=0.85)
    plt.title("Distribusi Target Biner (Rasio Imbalance)", fontsize=11, fontweight="bold")
    plt.ylabel("Jumlah Mahasiswa")
    for i, v in enumerate(bin_counts.values):
        plt.text(i, v + 20, f"{v} ({v/len(y)*100:.1f}%)", ha="center", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=200)
    plt.close()
    print("[OK] Figure 1 tersimpan: class_distribution.png")

    # -------------------------------------------------------------
    # 2. PEMBAGIAN DATA (TRAIN / TEST SPLIT TERSTRATIFIKASI)
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain Set: {X_train.shape[0]} sampel | Test Set: {X_test.shape[0]} sampel")

    # -------------------------------------------------------------
    # 3. DEFINISI MODEL EVALUASI
    # -------------------------------------------------------------
    models = {
        "Logistic_Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced", random_state=42
        ),
        "Gradient_Boosting (Proposed)": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        )
    }

    results = {}
    test_predictions = {}
    
    print("\n" + "=" * 70)
    print("4. PELATIHAN DAN PENGUKURAN EVALUATION METRICS PADA TEST SET")
    print("=" * 70)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1_bin = f1_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision (Dropout)": round(float(prec), 4),
            "Recall / Sensitivity": round(float(rec), 4),
            "F1-Score (Dropout)": round(float(f1_bin), 4),
            "Macro F1-Score": round(float(f1_macro), 4),
            "ROC-AUC": round(float(roc_auc), 4),
            "PR-AUC (Average Precision)": round(float(pr_auc), 4),
            "Confusion_Matrix": {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1])
            }
        }
        
        test_predictions[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "cm": cm
        }
        
        print(f"\nModel: {name}")
        print(f" - Accuracy     : {acc:.4f}")
        print(f" - Precision    : {prec:.4f}")
        print(f" - Recall (Sens): {rec:.4f}")
        print(f" - F1-Score (Bin): {f1_bin:.4f} | Macro-F1: {f1_macro:.4f}")
        print(f" - ROC-AUC      : {roc_auc:.4f}")
        print(f" - PR-AUC       : {pr_auc:.4f}")
        print(f" - Confusion Matrix: [TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}]")

    # Simpan hasil metrik ke JSON
    with open(os.path.join(processed_dir, "metrics_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[OK] Rangkuman metrik evaluasi tersimpan di: data/processed/metrics_summary.json")

    # -------------------------------------------------------------
    # Visualisasi 2: Confusion Matrices
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, pred_info) in enumerate(test_predictions.items()):
        cm = pred_info["cm"]
        ax = axes[i]
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                lbl = "TN" if (r==0 and c==0) else "FP" if (r==0 and c==1) else "FN" if (r==1 and c==0) else "TP"
                ax.text(c, r, f"{lbl}\n{val}", va="center", ha="center", fontsize=11, fontweight="bold",
                        color="white" if val > cm.max()/2 else "black")
        clean_name = name.split(" ")[0].replace("_", " ")
        ax.set_title(f"{clean_name}", fontsize=11, pad=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Dropout", "Dropout"])
        ax.set_yticklabels(["Non-Dropout", "Dropout"])
        ax.set_xlabel("Predicted Label")
        if i == 0:
            ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrices.png"), dpi=200)
    plt.close()
    print("[OK] Figure 2 tersimpan: confusion_matrices.png")

    # -------------------------------------------------------------
    # Visualisasi 3: Kurva ROC dan Kurva Precision-Recall
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot ROC Curves
    for name, pred_info in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, pred_info["y_proba"])
        auc_val = results[name]["ROC-AUC"]
        clean_name = name.split(" ")[0].replace("_", " ")
        axes[0].plot(fpr, tpr, label=f"{clean_name} (AUC = {auc_val:.3f})", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=9)

    # Plot PR Curves
    for name, pred_info in test_predictions.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, pred_info["y_proba"])
        prauc_val = results[name]["PR-AUC (Average Precision)"]
        clean_name = name.split(" ")[0].replace("_", " ")
        axes[1].plot(recall_vals, precision_vals, label=f"{clean_name} (PR-AUC = {prauc_val:.3f})", lw=2)
    
    baseline_pr = y_test.sum() / len(y_test)
    axes[1].axhline(y=baseline_pr, color="r", linestyle="--", alpha=0.5, label=f"Baseline Prevalensi ({baseline_pr:.2f})")
    axes[1].set_title("Precision-Recall Curve (PR-AUC)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall (Coverage Siswa Berisiko)")
    axes[1].set_ylabel("Precision (Akurasi Peringatan Dini)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_pr_curves.png"), dpi=200)
    plt.close()
    print("[OK] Figure 3 tersimpan: roc_pr_curves.png")

    # -------------------------------------------------------------
    # Visualisasi 4: Permutation Feature Importance (Aproksimasi XAI)
    # -------------------------------------------------------------
    gb_model = models["Gradient_Boosting (Proposed)"]
    perm_importance = permutation_importance(gb_model, X_test, y_test, n_repeats=10, random_state=42)
    
    sorted_idx = perm_importance.importances_mean.argsort()[::-1][:10]
    top_features = [features[i] for i in sorted_idx]
    top_importances = perm_importance.importances_mean[sorted_idx]
    
    plt.figure(figsize=(9, 5))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1], color="#2c7fb8", edgecolor="black", alpha=0.85)
    plt.yticks(y_pos, [f.replace("_", " ") for f in top_features[::-1]], fontsize=9)
    plt.xlabel("Permutation Feature Importance (Mean Decrease in PR-AUC / Accuracy)")
    plt.title("Top 10 Indikator Utama Deteksi Dini Drop-Out (Model Gradient Boosting)", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=200)
    plt.close()
    print("[OK] Figure 4 tersimpan: feature_importance.png")
    
    print("\n" + "=" * 70)
    print("EKSPERIMEN & VISUALISASI SELESAI DENGAN SUKSES!")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
