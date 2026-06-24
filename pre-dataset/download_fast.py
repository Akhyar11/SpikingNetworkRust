import json
import os
import concurrent.futures
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

languages = ['id', 'es', 'fr', 'de', 'zh', 'ar', 'ru', 'pt', 'ja', 'ko', 'hi', 'it', 'tr', 'vi', 'th', 'fa', 'pl', 'uk', 'nl']

def fetch_lang(lang):
    pair = f"en-{lang}" if "en" < lang else f"{lang}-en"
    print(f"Downloading {pair}...")
    try:
        # Load exactly 55000 rows without streaming the whole dataset
        # by using the dataset slicing (this downloads the parquet file but then slices it)
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split="train[:55000]")
        
        train_res = []
        eval_res = []
        
        for i, row in enumerate(ds):
            item = {"en": row["translation"]["en"], lang: row["translation"][lang]}
            if i < 50000:
                train_res.append(item)
            else:
                eval_res.append(item)
        return lang, train_res, eval_res
    except Exception as e:
        print(f"Error {pair}: {e}")
        return lang, [], []

train_all = []
eval_all = []

print("Starting parallel download...")
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetch_lang, languages)
    
    for lang, t_res, e_res in results:
        train_all.extend(t_res)
        eval_all.extend(e_res)
        print(f"Finished processing {lang}. Total train: {len(train_all)}")

with open("dataset_multilingual_train.json", "w", encoding="utf-8") as f:
    json.dump(train_all, f, ensure_ascii=False, indent=2)
with open("dataset_multilingual_eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_all, f, ensure_ascii=False, indent=2)

print("Done successfully!")
