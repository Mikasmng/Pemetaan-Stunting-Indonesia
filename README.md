# Pemetaan Stunting Indonesia
Dashboard analitik untuk visualisasi dan analisis prevalensi stunting di Indonesia berdasarkan data Survei Status Gizi Indonesia (SSGI) 2024.
Dashboard menampilkan pemetaan pada tingkat provinsi berdasarkan agregasi data kabupaten/kota, serta menyediakan informasi prevalensi stunting, indikator pendukung, hasil klasifikasi menggunakan Extreme Gradient Boosting (XGBoost), evaluasi model, feature importance, dan kalkulator antropometri WHO.
Dataset yang digunakan mencakup 514 kabupaten/kota dari 38 provinsi. Pada visualisasi peta, data kabupaten/kota dikelompokkan berdasarkan provinsi sehingga dapat memberikan gambaran kondisi stunting antarprovinsi.

## Fitur Utama
- Peta persebaran prevalensi stunting menurut provinsi
- Data prevalensi stunting kabupaten/kota
- Filter berdasarkan provinsi dan kategori prevalensi
- Prediksi kategori menggunakan XGBoost
- Probabilitas hasil prediksi model
- Evaluasi model dan confusion matrix
- Feature importance
- Visualisasi indikator wilayah
- Kalkulator antropometri WHO

## Sumber Data
Data berasal dari publikasi **Survei Status Gizi Indonesia (SSGI) 2024 Dalam Angka** yang diterbitkan oleh Kementerian Kesehatan Republik Indonesia.

## Demo Dashboard
Dashboard dapat diakses melalui:
https://pemetaan-stunting-indonesia-dashboa.vercel.app/

## Catatan
Pemetaan tingkat provinsi menggunakan agregasi atau rata-rata data kabupaten/kota dalam masing-masing provinsi. Nilai tersebut tidak dimaksudkan sebagai prevalensi resmi provinsi.
