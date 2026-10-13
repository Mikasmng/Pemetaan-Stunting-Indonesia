import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "dataset_risiko_stunting_kabkota.csv"
OUTPUT_PREDIKSI = BASE_DIR / "data" / "hasil_prediksi_model.csv"
OUTPUT_MODEL_JSON = BASE_DIR / "dashboard" / "model_output.json"

# Satu fitur kabupaten/kota memberi variasi intra-provinsi, sedangkan tiga fitur
# kontekstual tetap dipertahankan sesuai fokus awal penelitian.
FITUR = [
    "underweight_pct",
    "akses_air_minum_layak_prov_pct",
    "akses_sanitasi_aman_prov_pct",
    "pemantauan_pertumbuhan_sesuai_standar_prov_pct",
]
FITUR_KABKOTA = ["underweight_pct"]
FITUR_PROVINSI = [
    "akses_air_minum_layak_prov_pct",
    "akses_sanitasi_aman_prov_pct",
    "pemantauan_pertumbuhan_sesuai_standar_prov_pct",
]
TARGET = "kategori_prevalensi_stunting"
SEMUA_KATEGORI_STANDAR = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
KATEGORI_OPERASIONAL = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
N_ITER_TUNING = 60


def kategori_prevalensi_stunting(pct):
    """Kategori prevalensi stunting WHO/UNICEF (de Onis et al., 2019)."""
    if pct < 2.5:
        return "Sangat Rendah"
    if pct < 10:
        return "Rendah"
    if pct < 20:
        return "Sedang"
    if pct < 30:
        return "Tinggi"
    return "Sangat Tinggi"


def python_type(obj):
    """Konversi tipe NumPy agar dapat disimpan ke JSON."""
    if isinstance(obj, dict):
        return {str(k): python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [python_type(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def hitung_metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
    }


# -----------------------------------------------------------------------------
# 1. Membaca dan memvalidasi dataset
# -----------------------------------------------------------------------------
df = pd.read_csv(DATASET)
kolom_wajib = ["kabupaten_kota", "provinsi", "stunting_pct"] + FITUR
kolom_hilang = [kolom for kolom in kolom_wajib if kolom not in df.columns]
if kolom_hilang:
    raise ValueError(f"Kolom wajib tidak ditemukan di dataset: {kolom_hilang}")

# Target dibentuk ulang dari stunting_pct agar konsisten dengan definisi proposal.
df[TARGET] = df["stunting_pct"].apply(kategori_prevalensi_stunting)
df.to_csv(DATASET, index=False)

X = df[FITUR].copy()
y_teks = df[TARGET].copy()
groups = df["provinsi"].copy()

# Urutan kelas dibuat mengikuti tingkat prevalensi, bukan urutan alfabet.
kelas_model = [k for k in SEMUA_KATEGORI_STANDAR if k in set(y_teks)]
kelas_ke_indeks = {kelas: i for i, kelas in enumerate(kelas_model)}
indeks_ke_kelas = np.array(kelas_model, dtype=object)
y = y_teks.map(kelas_ke_indeks).to_numpy(dtype=int)
labels = list(range(len(kelas_model)))

print("=== PEMODELAN XGBOOST SSGI 2024 - REVISI 2 ===")
print("Fitur kabupaten/kota:", ", ".join(FITUR_KABKOTA))
print("Fitur kontekstual provinsi:", ", ".join(FITUR_PROVINSI))
print("Kategori model:", kelas_model)
print("Distribusi target:")
print(y_teks.value_counts())
print("Missing value pada setiap fitur:")
print(X.isna().sum())

# -----------------------------------------------------------------------------
# 2. Pipeline dan ruang pencarian hyperparameter
# -----------------------------------------------------------------------------
model_dasar = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(kelas_model),
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    tree_method="hist",
    n_jobs=1,
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", model_dasar),
])

parameter_space = {
    "model__max_depth": [1, 2, 3, 4],
    "model__learning_rate": [0.015, 0.03, 0.05, 0.08, 0.1],
    "model__n_estimators": [100, 150, 200, 300, 400, 500],
    "model__min_child_weight": [1, 2, 3, 5, 8],
    "model__gamma": [0, 0.05, 0.1, 0.25, 0.5],
    "model__subsample": [0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.8, 0.9, 1.0],
    "model__reg_alpha": [0, 0.01, 0.05, 0.1, 0.5],
    "model__reg_lambda": [1, 2, 5, 10],
}

f1_macro_scorer = make_scorer(
    f1_score,
    average="macro",
    labels=labels,
    zero_division=0,
)

# -----------------------------------------------------------------------------
# 3. Nested Stratified Group Cross-Validation
#    Outer fold menghasilkan prediksi out-of-fold untuk evaluasi yang lebih
#    stabil. Inner fold hanya digunakan untuk memilih hyperparameter.
# -----------------------------------------------------------------------------
outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_pred = np.full(len(df), -1, dtype=int)
oof_proba = np.full((len(df), len(kelas_model)), np.nan, dtype=float)
baseline_oof_pred = np.full(len(df), -1, dtype=int)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups), start=1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=100 + fold,
    )

    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=N_ITER_TUNING,
        scoring=f1_macro_scorer,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        random_state=1000 + fold,
        error_score=0,
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
        model__sample_weight=sample_weight_train,
    )

    pred_fold = search.best_estimator_.predict(X_test)
    proba_fold = search.best_estimator_.predict_proba(X_test)

    oof_pred[test_idx] = pred_fold
    oof_proba[test_idx, :] = proba_fold

    # Baseline majority-class pada split yang sama sebagai pembanding minimum.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train_idx), 1)), y_train)
    baseline_oof_pred[test_idx] = baseline.predict(np.zeros((len(test_idx), 1)))

    fold_metrics = hitung_metrics(y_test, pred_fold, labels)
    fold_results.append({
        "fold": fold,
        "jumlah_data_uji": int(len(test_idx)),
        "jumlah_provinsi_uji": int(groups_test.nunique()),
        "provinsi_uji": sorted(groups_test.unique().tolist()),
        "f1_macro": fold_metrics["f1_macro"],
        "accuracy": fold_metrics["accuracy"],
        "inner_cv_best_f1_macro": float(search.best_score_),
        "best_params": search.best_params_,
    })

    print(
        f"Fold {fold}: F1 Macro={fold_metrics['f1_macro']:.4f}, "
        f"Accuracy={fold_metrics['accuracy']:.4f}, "
        f"Provinsi uji={groups_test.nunique()}"
    )

if np.any(oof_pred < 0) or np.isnan(oof_proba).any():
    raise RuntimeError("Prediksi out-of-fold tidak lengkap.")

# -----------------------------------------------------------------------------
# 4. Evaluasi agregat out-of-fold
# -----------------------------------------------------------------------------
metrics = hitung_metrics(y, oof_pred, labels)
baseline_metrics = hitung_metrics(y, baseline_oof_pred, labels)
report = classification_report(
    y,
    oof_pred,
    labels=labels,
    target_names=kelas_model,
    zero_division=0,
    output_dict=True,
)
cm = confusion_matrix(y, oof_pred, labels=labels)

fold_f1 = [x["f1_macro"] for x in fold_results]
fold_accuracy = [x["accuracy"] for x in fold_results]

print("\n=== EVALUASI OUT-OF-FOLD ===")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Precision Macro: {metrics['precision_macro']:.4f}")
print(f"Recall Macro: {metrics['recall_macro']:.4f}")
print(f"F1 Macro: {metrics['f1_macro']:.4f}")
print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
print(f"Rata-rata F1 Macro antar-fold: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.2%}")
print(f"Baseline F1 Macro: {baseline_metrics['f1_macro']:.4f}")
print("\nClassification Report:")
print(classification_report(y, oof_pred, labels=labels, target_names=kelas_model, zero_division=0))
print("Confusion Matrix:")
print(cm)

# -----------------------------------------------------------------------------
# 5. Tuning model final pada seluruh dataset
#    Ini hanya untuk model akhir/feature importance. Metrik evaluasi di atas
#    tetap berasal dari prediksi out-of-fold, bukan dari model full-data.
# -----------------------------------------------------------------------------
final_cv = StratifiedGroupKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=2024,
)
final_sample_weight = compute_sample_weight(class_weight="balanced", y=y)

final_search = RandomizedSearchCV(
    estimator=clone(pipeline),
    param_distributions=parameter_space,
    n_iter=N_ITER_TUNING,
    scoring=f1_macro_scorer,
    cv=final_cv,
    n_jobs=-1,
    refit=True,
    random_state=2024,
    error_score=0,
)
final_search.fit(
    X,
    y,
    groups=groups,
    model__sample_weight=final_sample_weight,
)
model_final = final_search.best_estimator_

# Prediksi yang ditampilkan untuk baris 2024 adalah prediksi OOF agar setiap
# observasi dinilai oleh model yang tidak dilatih pada provinsinya sendiri.
probabilitas_oof = oof_proba.max(axis=1)
hasil_prediksi = df[["kabupaten_kota", "provinsi", "stunting_pct", TARGET]].copy()
hasil_prediksi["prediksi_model"] = indeks_ke_kelas[oof_pred]
hasil_prediksi["probabilitas_model"] = np.round(probabilitas_oof, 4)
hasil_prediksi["jenis_prediksi"] = "out-of-fold"
hasil_prediksi.to_csv(OUTPUT_PREDIKSI, index=False)

feature_importance_values = model_final.named_steps["model"].feature_importances_
feature_importance = {
    fitur: round(float(nilai), 6)
    for fitur, nilai in zip(FITUR, feature_importance_values)
}

# -----------------------------------------------------------------------------
# 6. Simpan hasil model ke JSON untuk dashboard
# -----------------------------------------------------------------------------
model_output = {
    "tahun_data": 2024,
    "algoritma": "XGBoost Classifier",
    "revisi_model": 2,
    "unit_analisis": (
        "target kategori prevalensi stunting pada kabupaten/kota; "
        "underweight berasal dari kabupaten/kota dan tiga fitur kontekstual berasal dari provinsi"
    ),
    "unit_target": "kabupaten/kota",
    "unit_fitur": {
        "kabupaten_kota": FITUR_KABKOTA,
        "provinsi": FITUR_PROVINSI,
    },
    "jumlah_data": int(len(df)),
    "jumlah_provinsi": int(groups.nunique()),
    "fitur": FITUR,
    "kelas_model": kelas_model,
    "kategori_standar": SEMUA_KATEGORI_STANDAR,
    "kategori_operasional": KATEGORI_OPERASIONAL,
    "evaluation_method": "Nested StratifiedGroupKFold 5 outer folds dengan tuning 4 inner folds",
    "jumlah_evaluasi_oof": int(len(df)),
    "preprocessing": {
        "missing_value": "SimpleImputer median di dalam pipeline; fit hanya pada data training setiap fold",
        "normalisasi": "Tidak digunakan karena XGBoost berbasis pohon",
        "outer_evaluation": "StratifiedGroupKFold 5 fold berbasis provinsi",
        "inner_tuning": "StratifiedGroupKFold 4 fold berbasis provinsi",
        "class_imbalance": "sample_weight balanced dihitung pada data training",
        "scoring_tuning": "F1 Macro",
    },
    "metrics": {k: round(v, 6) for k, v in metrics.items()},
    "baseline_metrics": {k: round(v, 6) for k, v in baseline_metrics.items()},
    "fold_f1_macro": [round(float(v), 6) for v in fold_f1],
    "fold_accuracy": [round(float(v), 6) for v in fold_accuracy],
    "fold_f1_macro_mean": round(float(np.mean(fold_f1)), 6),
    "fold_f1_macro_std": round(float(np.std(fold_f1)), 6),
    "fold_results": fold_results,
    "final_tuning_best_params": final_search.best_params_,
    "final_tuning_cv_f1_macro": round(float(final_search.best_score_), 6),
    # alias agar dashboard lama tetap dapat membaca parameter.
    "best_params": final_search.best_params_,
    "cv_f1_macro": round(float(np.mean(fold_f1)), 6),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "feature_importance": feature_importance,
    "prediction_display": "out-of-fold",
    "underweight_role": "indikator status gizi pendamping tingkat kabupaten/kota yang digunakan sebagai fitur prediktif; bukan determinan dan bukan bukti hubungan sebab-akibat terhadap stunting",
    "feature_importance_source": "model final full-data yang dilatih setelah evaluasi out-of-fold selesai",
    "evaluation_source": "metrik, classification report, dan confusion matrix berasal dari prediksi out-of-fold berbasis provinsi",
    "catatan": [
        "Seluruh metrik evaluasi berasal dari prediksi out-of-fold; setiap kabupaten/kota diprediksi oleh model yang tidak dilatih pada provinsinya sendiri.",
        "Hyperparameter dipilih pada inner cross-validation, sedangkan outer cross-validation digunakan untuk evaluasi.",
        "Underweight merupakan indikator status gizi pendamping tingkat kabupaten/kota yang digunakan sebagai fitur prediktif; variabel ini tidak diposisikan sebagai determinan atau faktor penyebab stunting.",
        "Tiga indikator kontekstual tetap berasal dari tingkat provinsi sehingga interpretasinya tetap bersifat kontekstual.",
        "Feature importance berasal dari model final full-data setelah evaluasi out-of-fold selesai; nilainya menunjukkan kontribusi fitur pada model dan tidak membuktikan hubungan sebab-akibat.",
        "Kategori Sangat Rendah (<2,5%) tetap merupakan ambang standar WHO-UNICEF, tetapi tidak menjadi kategori operasional karena tidak terdapat observasi pada dataset 2024.",
        "Analisis operasional dan model menggunakan empat kelas teramati: Rendah, Sedang, Tinggi, dan Sangat Tinggi.",
    ],
}

OUTPUT_MODEL_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_MODEL_JSON, "w", encoding="utf-8") as f:
    json.dump(python_type(model_output), f, ensure_ascii=False, indent=2)

print("\nParameter model final:", final_search.best_params_)
print("F1 Macro tuning model final:", round(float(final_search.best_score_), 4))
print("\nFeature importance model final:")
for fitur, nilai in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {fitur}: {nilai:.4f}")
print("\nPrediksi OOF disimpan ke:", OUTPUT_PREDIKSI)
print("Hasil evaluasi model disimpan ke:", OUTPUT_MODEL_JSON)
