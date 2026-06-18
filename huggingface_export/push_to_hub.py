import os
from transformers import AutoTokenizer
from modeling_spiking import SpikingConfig, SpikingSentenceEmbedder
import warnings

warnings.filterwarnings("ignore")

def push_model():
    # ---------------------------------------------------------
    # GANTI 'repo_id' DI BAWAH INI DENGAN NAMA REPOSITORI ANDA
    # Contoh: "Akhyar11/spiking-sentence-embedder-v1"
    # ---------------------------------------------------------
    repo_id = "PulseNet-Labs/spiking-sentence-embedder" 
    
    print("Memuat Tokenizer dan SNN Model dari lokal...")
    tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
    
    config = SpikingConfig.from_pretrained(".")
    model = SpikingSentenceEmbedder(config)
    
    # Simpan state_dict terbaru menggunakan standar HF (menghasilkan model.safetensors)
    model.save_pretrained(".")
    
    print(f"\nMengunggah ke Hugging Face Hub (Repositori: {repo_id})...")
    print("Mungkin akan memakan waktu beberapa saat tergantung ukuran model dan koneksi internet.")
    
    # 1. Unggah Tokenizer
    tokenizer.push_to_hub(repo_id)
    print("- Tokenizer berhasil diunggah!")
    
    # 2. Unggah Model, Config, dan Kode (modeling_spiking.py)
    model.push_to_hub(repo_id)
    print("- Model SNN & Arsitektur berhasil diunggah!")
    
    print("\nSelesai! Model sekarang tersedia di Hugging Face.")
    print(f"Tautan: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    push_model()
