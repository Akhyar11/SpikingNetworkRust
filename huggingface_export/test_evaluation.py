import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from modeling_spiking import SpikingSentenceEmbedder, SpikingConfig
import warnings

warnings.filterwarnings("ignore")

def pearson(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    if n == 0: return 0.0
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x * x)
    syy = np.sum(y * y)
    sxy = np.sum(x * y)
    
    num = n * sxy - sx * sy
    den = np.sqrt(max(0.0, n * sxx - sx * sx) * max(0.0, n * syy - sy * sy))
    if den == 0.0: return 0.0
    return num / den

def evaluate():
    print("Memuat Tokenizer dan Model dari folder huggingface_export...")
    tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
    config = SpikingConfig.from_pretrained(".")
    model = SpikingSentenceEmbedder(config)
    model.eval()

    dataset_path = "../experiment/file_model/all_sts_teacher_scored.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset tidak ditemukan di {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Berhasil memuat dataset evaluasi: {len(data)} pasangan kalimat.")
    print("Memulai proses evaluasi...\n")
    
    preds = []
    targets = []
    
    # Menguji pada seluruh data
    test_data = data
    
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            
            s1_list = [pair["sentence1"].lower() for pair in batch]
            s2_list = [pair["sentence2"].lower() for pair in batch]
            scores = [pair["score"] for pair in batch]
            
            inputs1 = tokenizer(s1_list, padding="max_length", max_length=model.max_seq_len, truncation=True, return_tensors="pt")
            inputs2 = tokenizer(s2_list, padding="max_length", max_length=model.max_seq_len, truncation=True, return_tensors="pt")
            
            # Ganti token 1 (PAD) menjadi 0 (UNK) karena di Rust array diinisialisasi dengan 0usize!
            inputs1.input_ids[inputs1.input_ids == tokenizer.pad_token_id] = 0
            inputs2.input_ids[inputs2.input_ids == tokenizer.pad_token_id] = 0
            
            embs1 = model(**inputs1)
            embs2 = model(**inputs2)
            

            
            # Mean-centering sebelum cosine similarity (Pearson similarity pada level vektor, sama dengan Rust)
            embs1_centered = embs1 - embs1.mean(dim=-1, keepdim=True)
            embs2_centered = embs2 - embs2.mean(dim=-1, keepdim=True)
            sims = F.cosine_similarity(embs1_centered, embs2_centered).clamp(min=0.0).tolist()
            
            preds.extend(sims)
            targets.extend(scores)
            
            if i % 5000 == 0:
                print(f"Progress: {i}/{len(test_data)}...")

    print("\n    --- Contoh 5 Prediksi PyTorch SNN vs Target Guru ---")
    for i in range(min(5, len(preds))):
        print(f"      S1: {test_data[i]['sentence1']}")
        print(f"      S2: {test_data[i]['sentence2']}")
        print(f"      PyTorch SNN Pred: {preds[i]:.4f} | Target Guru: {targets[i]:.4f}\n")
        
    r = pearson(preds, targets)
    print(f"==> Hasil Akhir Pearson Correlation (PyTorch): {r:.4f}")

if __name__ == "__main__":
    evaluate()
