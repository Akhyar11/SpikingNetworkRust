import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing

def convert():
    # 1. Baca vocab.json buatan Rust
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        rust_data = json.load(f)
        
    vocab = rust_data.get("vocab", {})
    # Rust menyimpannya sebagai array of strings [ ["a", "b"] ]
    # Tokenizers HF butuhnya string digabung: ["a b"]
    raw_merges = rust_data.get("merges", [])
    merges = [(m[0], m[1]) for m in raw_merges]

    # 2. Inisialisasi Model BPE bawaan Hugging Face Tokenizers
    unk_token = "<UNK>"
    # Buat objek BPE
    model = BPE(vocab, merges, unk_token=unk_token)
    
    # 3. Rakit Tokenizer
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    
    # Tambahkan Special Tokens
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    tokenizer.add_special_tokens(special_tokens)
    
    # 4. Simpan kembali sebagai format tokenizer.json standar Hugging Face
    tokenizer.save("tokenizer.json")
    print("Berhasil mengonversi Tokenizer Rust ke format standar Hugging Face!")

if __name__ == "__main__":
    convert()
