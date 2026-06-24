import json
import os
from datasets import load_dataset
import concurrent.futures

# 19 bahasa + Inggris = 20 bahasa utama
languages = ['id', 'es', 'fr', 'de', 'zh', 'ar', 'ru', 'pt', 'ja', 'ko', 'hi', 'it', 'tr', 'vi', 'th', 'fa', 'pl', 'uk', 'nl']

def fetch_lang(lang):
    pair = f"en-{lang}" if "en" < lang else f"{lang}-en"
    print(f"Mengambil data untuk pasangan: {pair}...")
    try:
        # Mengambil 55.000 baris pertama
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split="train[:55000]")
        train_res = []
        eval_res = []
        for i, row in enumerate(ds):
            item = {"en": row["translation"]["en"], lang: row["translation"][lang]}
            if i < 50000:
                train_res.append(item)
            else:
                eval_res.append(item)
        print(f"Selesai {pair}: 50k train, 5k eval.")
        return train_res, eval_res
    except Exception as e:
        print(f"Gagal mengambil {pair}: {e}")
        return [], []

train_all = []
eval_all = []

print("Mulai mendownload dataset untuk 20 bahasa (menggunakan OPUS-100 sebagai alternatif NLLB)...")
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(fetch_lang, languages)
    for t_res, e_res in results:
        train_all.extend(t_res)
        eval_all.extend(e_res)

with open("dataset_20_bahasa_train.json", "w", encoding="utf-8") as f:
    json.dump(train_all, f, ensure_ascii=False, indent=2)
with open("dataset_20_bahasa_eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_all, f, ensure_ascii=False, indent=2)

print(f"Selesai! Total Train: {len(train_all)}, Total Eval: {len(eval_all)}")
