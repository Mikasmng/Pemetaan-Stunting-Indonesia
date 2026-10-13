import json

with open("dashboard/provinsi.geojson", encoding="utf-8") as f:
    geo = json.load(f)

print("Jumlah provinsi (features):", len(geo["features"]))

print("\nContoh properti provinsi pertama:")
print(geo["features"][0]["properties"])

print("\nDaftar semua nama provinsi di file GeoJSON:")
for f in geo["features"]:
    print("-", f["properties"])

print("\nTipe geometry provinsi pertama:", geo["features"][0]["geometry"]["type"])
coords = geo["features"][0]["geometry"]["coordinates"]
print("\nContoh koordinat mentah (dipotong):")
print(coords[0][0][:2] if isinstance(coords[0][0], list) else coords[:2])