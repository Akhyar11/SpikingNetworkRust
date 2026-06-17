import urllib.request
import zipfile
import json
import os
import random

print("=====================================================")
print("  MENDOWNLOAD DATASET SNLI (STANFORD)")
print("=====================================================")

url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
zip_path = "snli_1.0.zip"

if not os.path.exists(zip_path):
    print("Mendownload file zip SNLI (sekitar 90MB)...")
    urllib.request.urlretrieve(url, zip_path)
else:
    print("File zip sudah ada, langsung mengekstrak...")

print("Mengekstrak file dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Hanya mengekstrak file train yang dibutuhkan agar tidak menuhin memori
    zip_ref.extract("snli_1.0/snli_1.0_train.jsonl", path=".")

print("Memproses dataset dan mengubah label ke skor angka (0.0 - 1.0)...")
data_out = []
with open("snli_1.0/snli_1.0_train.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        label = row.get('gold_label')
        
        # Abaikan label yang tidak jelas
        if label == '-':
            continue
            
        score = 0.0
        if label == 'entailment':
            score = 1.0     # Entailment -> 1.0
        elif label == 'neutral':
            score = 0.5     # Neutral -> 0.5
        elif label == 'contradiction':
            score = 0.0     # Contradiction -> 0.0
            
        s1 = row.get('sentence1', '')
        s2 = row.get('sentence2', '')
        
        if not s1 or not s2:
            continue
            
        data_out.append({
            "s1": s1,
            "s2": s2,
            "score": score
        })

# Acak agar training merata
random.seed(42)
random.shuffle(data_out)

# Ambil 250.000 pasang agar setara dengan 2.5x dataset lama Anda dan waktu training ideal
limit = 250000
data_out = data_out[:limit]

out_path = 'experiment/file_model/teacher_distillation_dataset.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

print(f"Menyimpan {len(data_out):,} pasang kalimat ke {out_path}...")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(data_out, f, ensure_ascii=False, indent=2)

# Bersihkan file temp agar rapi
try:
    os.remove("snli_1.0/snli_1.0_train.jsonl")
    os.rmdir("snli_1.0")
    os.remove(zip_path)
except:
    pass

print("Selesai! Dataset berkualitas tinggi (SNLI) sudah berhasil di-generate!")
print("Anda bisa langsung menjalankan cargo run --bin train_distil_only")
