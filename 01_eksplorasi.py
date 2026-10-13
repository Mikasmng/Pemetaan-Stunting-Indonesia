from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "dataset_risiko_stunting_kabkota.csv"


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


df = pd.read_csv(DATASET)

# Bentuk ulang kategori prevalensi agar selalu konsisten dengan metodologi proposal.
df["kategori_prevalensi_stunting"] = df["stunting_pct"].apply(kategori_prevalensi_stunting)

print("=== EKSPLORASI DATA SSGI 2024 ===")
print("Jumlah baris & kolom:", df.shape)
print("Jumlah provinsi:", df["provinsi"].nunique())
print("Jumlah kabupaten/kota:", len(df))
print("\nNama kolom:")
print(df.columns.tolist())
print("\nJumlah nilai kosong per kolom:")
print(df.isna().sum())
print("\nDistribusi kategori prevalensi stunting:")
print(df["kategori_prevalensi_stunting"].value_counts())
print("\nRata-rata aritmetika prevalensi kabupaten/kota:", round(df["stunting_pct"].mean(), 1), "%")
print("Prevalensi nasional resmi SSGI 2024: 19.8 %")
print("\nCatatan: rata-rata kabupaten/kota tidak sama dengan prevalensi nasional resmi SSGI.")
print("\nCatatan unit fitur: tiga prediktor adalah indikator tingkat provinsi yang dipadankan ke kabupaten/kota.")
print("\n5 baris pertama:")
print(df.head())
