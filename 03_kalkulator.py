# Kalkulator Z-score Panjang/Tinggi Badan menurut Umur (PB/U atau TB/U)
# Acuan: WHO Child Growth Standards (2006)
# 0-23 bulan  : panjang badan menurut umur (diukur berbaring)
# 24-60 bulan : tinggi badan menurut umur (diukur berdiri)
# Pada tabel LMS yang digunakan, L = 1 sehingga rumus:
# Z = (ukuran / M - 1) / S

WHO_LHFA = {
    "laki-laki": {
        "M": [49.8842,54.7244,58.4249,61.4292,63.8860,65.9026,67.6236,69.1645,70.5994,71.9687,73.2812,74.5388,75.7488,76.9186,78.0497,79.1458,80.2113,81.2487,82.2587,83.2418,84.1996,85.1348,86.0477,86.9410,87.1161,87.9720,88.8065,89.6197,90.4120,91.1828,91.9327,92.6631,93.3753,94.0711,94.7532,95.4236,96.0835,96.7337,97.3749,98.0073,98.6310,99.2459,99.8515,100.4485,101.0374,101.6186,102.1933,102.7625,103.3273,103.8886,104.4473,105.0041,105.5596,106.1138,106.6668,107.2188,107.7697,108.3198,108.8689,109.4170,109.9638],
        "S": [0.03795,0.03557,0.03424,0.03328,0.03257,0.03204,0.03165,0.03139,0.03124,0.03117,0.03118,0.03125,0.03137,0.03154,0.03174,0.03197,0.03222,0.03250,0.03279,0.03310,0.03342,0.03376,0.03410,0.03445,0.03507,0.03542,0.03576,0.03610,0.03642,0.03674,0.03704,0.03733,0.03761,0.03787,0.03812,0.03836,0.03858,0.03879,0.03900,0.03919,0.03937,0.03954,0.03971,0.03986,0.04002,0.04016,0.04031,0.04045,0.04059,0.04073,0.04086,0.04100,0.04113,0.04126,0.04139,0.04152,0.04165,0.04177,0.04190,0.04202,0.04214],
    },
    "perempuan": {
        "M": [49.1477,53.6872,57.0673,59.8029,62.0899,64.0301,65.7311,67.2873,68.7498,70.1435,71.4818,72.7710,74.0150,75.2176,76.3817,77.5099,78.6055,79.6710,80.7079,81.7182,82.7036,83.6654,84.6040,85.5202,85.7153,86.5904,87.4462,88.2830,89.1004,89.8991,90.6797,91.4430,92.1906,92.9239,93.6444,94.3533,95.0515,95.7399,96.4187,97.0885,97.7493,98.4015,99.0448,99.6795,100.3058,100.9238,101.5337,102.1360,102.7312,103.3197,103.9021,104.4786,105.0494,105.6148,106.1748,106.7295,107.2788,107.8227,108.3613,108.8948,109.4233],
        "S": [0.03790,0.03640,0.03568,0.03520,0.03486,0.03463,0.03448,0.03441,0.03440,0.03444,0.03452,0.03464,0.03479,0.03496,0.03514,0.03534,0.03555,0.03576,0.03598,0.03620,0.03643,0.03666,0.03688,0.03711,0.03764,0.03786,0.03808,0.03830,0.03851,0.03872,0.03893,0.03913,0.03933,0.03952,0.03971,0.03989,0.04006,0.04024,0.04041,0.04057,0.04073,0.04089,0.04105,0.04120,0.04135,0.04150,0.04164,0.04179,0.04193,0.04206,0.04220,0.04233,0.04246,0.04259,0.04272,0.04285,0.04298,0.04310,0.04322,0.04334,0.04347],
    },
}


def hitung_zscore_tbu(jenis_kelamin, umur_bulan, ukuran_cm):
    jenis_kelamin = jenis_kelamin.strip().lower()
    if jenis_kelamin not in WHO_LHFA:
        raise ValueError("jenis_kelamin harus 'laki-laki' atau 'perempuan'")
    if not isinstance(umur_bulan, int) or not 0 <= umur_bulan <= 60:
        raise ValueError("umur_bulan harus bilangan bulat antara 0 dan 60")
    if ukuran_cm <= 0:
        raise ValueError("ukuran_cm harus lebih dari 0")

    M = WHO_LHFA[jenis_kelamin]["M"][umur_bulan]
    S = WHO_LHFA[jenis_kelamin]["S"][umur_bulan]
    z = (ukuran_cm / M - 1) / S

    if z < -3:
        kategori = "Severely Stunted (sangat pendek)"
    elif z < -2:
        kategori = "Stunted (pendek)"
    elif z <= 3:
        kategori = "Normal"
    else:
        kategori = "Di atas rentang normal (perlu pemeriksaan lanjut)"

    indikator = "PB/U" if umur_bulan < 24 else "TB/U"
    cara_ukur = "berbaring" if umur_bulan < 24 else "berdiri"
    return round(z, 2), kategori, indikator, cara_ukur


def cek_input_manual():
    print("=== Kalkulator Z-score WHO PB/U atau TB/U (0-60 bulan) ===")
    jk = input("Jenis kelamin (laki-laki/perempuan): ").strip().lower()
    umur = int(input("Umur anak (bulan, 0-60): ").strip())
    ukuran = float(input("Panjang/tinggi badan (cm): ").strip())

    z, kategori, indikator, cara_ukur = hitung_zscore_tbu(jk, umur, ukuran)
    print(f"\nIndikator: {indikator} ({cara_ukur})")
    print(f"Z-score: {z}")
    print(f"Kategori: {kategori}")


if __name__ == "__main__":
    contoh = [
        ("laki-laki", 12, 75.7488),
        ("perempuan", 24, 85.7153),
        ("laki-laki", 60, 109.9638),
    ]
    print("=== Contoh Pengujian Otomatis ===")
    for jk, umur, ukuran in contoh:
        z, kategori, indikator, cara = hitung_zscore_tbu(jk, umur, ukuran)
        print(f"{jk}, {umur} bulan, {ukuran} cm -> {indikator}, Z={z}, {kategori}")
    print()
    cek_input_manual()
