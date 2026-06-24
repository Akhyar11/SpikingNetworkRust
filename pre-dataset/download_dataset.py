import json
import os
from datasets import load_dataset

# 19 languages + English = 20 languages
# All paired with English
languages = [
    'id', 'es', 'fr', 'de', 'zh', 'ar', 'ru', 'pt', 'ja', 'ko',
    'hi', 'it', 'tr', 'vi', 'th', 'fa', 'pl', 'uk', 'nl'
]

train_data = []
eval_data = []

print("Mulai mendownload dataset untuk 20 bahasa (19 pasangan dengan bahasa Inggris)...")

for lang in languages:
    # OPUS-100 language pairs are sorted alphabetically
    pair = f"en-{lang}" if "en" < lang else f"{lang}-en"
    print(f"Mengambil data untuk pasangan: {pair} ...")
    try:
        # Load stream to avoid downloading everything
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split="train", streaming=True)
        
        count = 0
        for row in ds:
            # format in opus-100: row['translation'] is a dict {'en': '...', lang: '...'}
            item = {
                "en": row["translation"]["en"],
                lang: row["translation"][lang]
            }
            if count < 50000:
                train_data.append(item)
            elif count < 55000:
                eval_data.append(item)
            else:
                break
            count += 1
            
    except Exception as e:
        print(f"Gagal mengambil {pair}: {e}")

print(f"Total train: {len(train_data)}, Total eval: {len(eval_data)}")

with open("dataset_multilingual_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
    
with open("dataset_multilingual_eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print("Selesai! Disimpan ke dataset_multilingual_train.json dan dataset_multilingual_eval.json")
