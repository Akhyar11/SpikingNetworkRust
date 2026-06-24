import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from modeling_spiking import SpikingConfig, SpikingSentenceEmbedder
import warnings

warnings.filterwarnings("ignore")

def push_model():
    # ---------------------------------------------------------
    # GANTI 'repo_id' DI BAWAH INI DENGAN NAMA REPOSITORI ANDA
    # ---------------------------------------------------------
    repo_id = "PulseNet-Labs/spiking-sentence-embedder-v2"

    print(f"Memuat Tokenizer dan SNN Model dari folder lokal...")
    tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)

    config = SpikingConfig.from_pretrained(".")
    model = SpikingSentenceEmbedder(config)

    # Pastikan model dalam mode eval
    model.eval()

    # Simpan state_dict sebagai model.safetensors (standar HF)
    print("Menyimpan model weights ke model.safetensors...")
    model.save_pretrained(".", safe_serialization=True)

    print(f"\nMengunggah ke Hugging Face Hub (Repositori: {repo_id})...")

    api = HfApi()

    # 1. Unggah Tokenizer
    print("- Mengunggah tokenizer...")
    tokenizer.push_to_hub(repo_id)
    print("  Tokenizer berhasil diunggah!")

    # 2. Unggah Config
    print("- Mengunggah config.json...")
    api.upload_file(
        path_or_fileobj="config.json",
        path_in_repo="config.json",
        repo_id=repo_id,
    )
    print("  config.json berhasil diunggah!")

    # 3. Unggah modeling_spiking.py (WAJIB untuk trust_remote_code)
    print("- Mengunggah modeling_spiking.py (custom architecture)...")
    api.upload_file(
        path_or_fileobj="modeling_spiking.py",
        path_in_repo="modeling_spiking.py",
        repo_id=repo_id,
    )
    print("  modeling_spiking.py berhasil diunggah!")

    # 4. Unggah model weights (safetensors) via API langsung
    print("- Mengunggah model.safetensors...")
    api.upload_file(
        path_or_fileobj="model.safetensors",
        path_in_repo="model.safetensors",
        repo_id=repo_id,
    )
    print("  Model weights berhasil diunggah!")

    print("\nSelesai! Model sekarang tersedia di Hugging Face.")
    print(f"Tautan: https://huggingface.co/{repo_id}")
    print("\nCara memuat model:")
    print(f'  from transformers import AutoTokenizer, AutoModel')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)')
    print(f'  model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)')


if __name__ == "__main__":
    push_model()
