import json
import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "dataset_risiko_stunting_kabkota.csv"
PREDIKSI = BASE_DIR / "data" / "hasil_prediksi_model.csv"
MODEL_JSON = BASE_DIR / "dashboard" / "model_output.json"
OUTPUT_JSON = BASE_DIR / "dashboard" / "data.json"

PREVALENSI_NASIONAL_RESMI_SSGI_2024 = 19.8
KATEGORI_STANDAR = ["Sangat Rendah", "Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]
KATEGORI_OPERASIONAL = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]


def kategori_prevalensi_stunting(pct):
    if pct < 2.5:
        return "Sangat Rendah"
    if pct < 10:
        return "Rendah"
    if pct < 20:
        return "Sedang"
    if pct < 30:
        return "Tinggi"
    return "Sangat Tinggi"


def aman_json(nilai):
    if nilai is None:
        return None
    if isinstance(nilai, (float, int)) and not math.isfinite(float(nilai)):
        return None
    return nilai


df = pd.read_csv(DATASET)
df["kategori_prevalensi_stunting"] = df["stunting_pct"].apply(kategori_prevalensi_stunting)

kolom_wajib = [
    "kabupaten_kota",
    "provinsi",
    "stunting_pct",
    "underweight_pct",
    "akses_air_minum_layak_prov_pct",
    "akses_sanitasi_aman_prov_pct",
    "pemantauan_pertumbuhan_sesuai_standar_prov_pct",
    "kategori_prevalensi_stunting",
]
kolom_hilang = [kolom for kolom in kolom_wajib if kolom not in df.columns]
if kolom_hilang:
    raise ValueError(f"Kolom wajib tidak ditemukan di dataset: {kolom_hilang}")

# Gabungkan prediksi model jika 02_model.py sudah dijalankan.
if PREDIKSI.exists():
    pred = pd.read_csv(PREDIKSI)[[
        "kabupaten_kota", "provinsi", "prediksi_model", "probabilitas_model"
    ]]
    df = df.merge(pred, on=["kabupaten_kota", "provinsi"], how="left")
else:
    df["prediksi_model"] = None
    df["probabilitas_model"] = None

# Data kabupaten/kota untuk tabel dan pencarian.
kabkota = []
for _, row in df.iterrows():
    kabkota.append({
        "kabupaten_kota": row["kabupaten_kota"],
        "provinsi": row["provinsi"],
        "stunting_pct": aman_json(row["stunting_pct"]),
        "underweight_pct": aman_json(row["underweight_pct"]),
        "kategori_prevalensi_stunting": row["kategori_prevalensi_stunting"],
        "prediksi_model": None if pd.isna(row["prediksi_model"]) else row["prediksi_model"],
        "probabilitas_model": None if pd.isna(row["probabilitas_model"]) else round(float(row["probabilitas_model"]), 4),
    })

# Rata-rata kabupaten/kota per provinsi. Ini bukan prevalensi resmi provinsi SSGI.
agg = df.groupby("provinsi", dropna=False).agg(
    rata_kabkota_stunting_pct=("stunting_pct", "mean"),
    akses_air_minum_layak_prov_pct=("akses_air_minum_layak_prov_pct", "mean"),
    akses_sanitasi_aman_prov_pct=("akses_sanitasi_aman_prov_pct", "mean"),
    pemantauan_pertumbuhan_sesuai_standar_prov_pct=("pemantauan_pertumbuhan_sesuai_standar_prov_pct", "mean"),
).round(1)

rata_provinsi = []
for provinsi, row in agg.sort_values("rata_kabkota_stunting_pct", ascending=False).iterrows():
    rata_provinsi.append({
        "provinsi": provinsi,
        # nama lama dipertahankan sebagai alias agar dashboard lama tidak langsung rusak
        "stunting_pct": aman_json(row["rata_kabkota_stunting_pct"]),
        "rata_kabkota_stunting_pct": aman_json(row["rata_kabkota_stunting_pct"]),
    })

peta_provinsi = {}
indikator_provinsi = {}
for provinsi, row in agg.iterrows():
    stunting = aman_json(row["rata_kabkota_stunting_pct"])
    air = aman_json(row["akses_air_minum_layak_prov_pct"])
    sanitasi = aman_json(row["akses_sanitasi_aman_prov_pct"])
    pemantauan = aman_json(row["pemantauan_pertumbuhan_sesuai_standar_prov_pct"])

    peta_provinsi[provinsi] = {
        "stunting_pct": stunting,
        "rata_kabkota_stunting_pct": stunting,
        "kategori": kategori_prevalensi_stunting(stunting),
        "keterangan": "Rata-rata aritmetika prevalensi kabupaten/kota dalam provinsi",
    }
    indikator_provinsi[provinsi] = {
        "akses_air_minum_layak_prov_pct": air,
        "akses_sanitasi_aman_prov_pct": sanitasi,
        "pemantauan_pertumbuhan_sesuai_standar_prov_pct": pemantauan,
    }

ringkasan_raw = df["kategori_prevalensi_stunting"].value_counts().to_dict()
ringkasan = {kategori: int(ringkasan_raw.get(kategori, 0)) for kategori in KATEGORI_OPERASIONAL}

model_output = None
if MODEL_JSON.exists():
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        model_output = json.load(f)

data_dashboard = {
    "tahun_data": 2024,
    "tahun_publikasi": 2025,
    "sumber_data": "Survei Status Gizi Indonesia (SSGI) 2024 Dalam Angka, Kementerian Kesehatan RI",
    "kabkota": kabkota,
    "rata_provinsi": rata_provinsi,
    "peta_provinsi": peta_provinsi,
    "indikator_provinsi": indikator_provinsi,
    "ringkasan": ringkasan,
    "kategori_standar": KATEGORI_STANDAR,
    "kategori_operasional": KATEGORI_OPERASIONAL,
    "total_kabkota": int(len(df)),
    "total_provinsi": int(df["provinsi"].nunique()),
    "prevalensi_nasional_resmi": PREVALENSI_NASIONAL_RESMI_SSGI_2024,
    "rata_kabkota": round(float(df["stunting_pct"].mean()), 1),
    "model": model_output,
    "catatan_data": [
        "Prevalensi nasional resmi SSGI 2024 adalah 19,8%.",
        f"Rata-rata kabupaten/kota dihitung secara aritmetika dari {len(df)} observasi dan bukan prevalensi nasional resmi.",
        "Nilai pada peta provinsi adalah rata-rata aritmetika kabupaten/kota dalam provinsi, bukan estimasi prevalensi resmi provinsi SSGI.",
        "Model XGBoost menggunakan underweight tingkat kabupaten/kota dan tiga indikator kontekstual tingkat provinsi.",
        "Tiga indikator kontekstual tingkat provinsi dipadankan ke kabupaten/kota sesuai provinsinya; underweight merupakan indikator status gizi pendamping tingkat kabupaten/kota yang digunakan sebagai fitur prediktif, bukan faktor penyebab stunting.",
        "Prediksi yang ditampilkan untuk data 2024 merupakan prediksi out-of-fold berbasis kelompok provinsi, bukan prediksi in-sample dari model yang dilatih pada baris yang sama.",
        "Metrik evaluasi dan confusion matrix berasal dari prediksi out-of-fold, sedangkan feature importance pada dashboard berasal dari model final full-data yang dilatih setelah evaluasi selesai.",
        "Kategori Sangat Rendah (<2,5%) tetap merupakan ambang standar WHO-UNICEF, tetapi tidak ditampilkan sebagai kategori operasional karena tidak terdapat observasi pada dataset 2024.",
        "Dashboard menggunakan empat kategori teramati: Rendah, Sedang, Tinggi, dan Sangat Tinggi.",
        "Tanda em dash pada dashboard menunjukkan data indikator tidak tersedia pada sumber yang digunakan.",
    ],
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data_dashboard, f, ensure_ascii=False, indent=2, allow_nan=False)

print(f"Berhasil! Data dashboard disimpan ke {OUTPUT_JSON}")
print("Tahun data:", data_dashboard["tahun_data"])
print("Jumlah kab/kota:", data_dashboard["total_kabkota"])
print("Jumlah provinsi:", data_dashboard["total_provinsi"])
print("Prevalensi nasional resmi SSGI 2024:", data_dashboard["prevalensi_nasional_resmi"], "%")
print("Rata-rata aritmetika kabupaten/kota:", data_dashboard["rata_kabkota"], "%")
print("Data model terintegrasi:", model_output is not None)
