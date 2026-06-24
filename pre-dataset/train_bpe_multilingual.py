"""
Train BPE tokenizer from multilingual validation dataset (dataset_20_bahasa_eval.json).
Output: experiment/file_model/vocab_multilingual.json
Compatible with src/core/bpe.rs (BPETokenizer::load)
"""
import json
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from collections import OrderedDict

# ===== CONFIG =====
DATASET_PATH = "dataset_20_bahasa_eval.json"
OUTPUT_PATH = "experiment/file_model/vocab_multilingual.json"
VOCAB_SIZE = 64_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]

def extract_texts(dataset_path):
    """Extract all text from all languages in the dataset."""
    print(f"📖 Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        for lang, text in item.items():
            if text and isinstance(text, str) and text.strip():
                texts.append(text)

    print(f"✅ Extracted {len(texts):,} text segments across {len(data):,} entries")
    return texts


def train_bpe(texts):
    """Train BPE tokenizer on texts."""
    print(f"🧠 Training BPE tokenizer (vocab_size={VOCAB_SIZE}, min_frequency={MIN_FREQUENCY})...")

    # Use BPE model with UNK token
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    # Normalizer: NFKC for Unicode normalization
    tokenizer.normalizer = normalizers.NFKC()

    # Pre-tokenizer: Metaspace (matches bpe.rs behavior)
    # bpe.rs splits on whitespace and prepends ▁ (U+2581) to each word.
    # Metaspace does exactly that: replaces spaces with ▁, keeping actual Unicode
    # characters (not byte-level encoding).
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
        split=True,
    )

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Train in batches to show progress
    def batch_iterator(texts, batch_size=100_000):
        for i in range(0, len(texts), batch_size):
            yield texts[i:i + batch_size]

    tokenizer.train_from_iterator(
        batch_iterator(texts),
        trainer=trainer,
    )

    print("✅ Training complete!")
    return tokenizer


def convert_to_bpe_rs_format(tokenizer):
    """
    Convert HuggingFace tokenizer to format expected by bpe.rs:
    { vocab: {token: id, ...}, merges: [[left, right], ...], config: {...} }

    Uses Metaspace pre-tokenizer with ▁ (U+2581) as word boundary marker,
    matching bpe.rs's WORD_BOUNDARY exactly.
    """
    # Serialize tokenizer to get merges
    serialized = json.loads(tokenizer.to_str())
    hf_vocab = serialized["model"]["vocab"]
    hf_merges = serialized["model"]["merges"]  # list of [left, right] lists

    # Build the custom vocab (Metaspace already uses ▁, no mapping needed)
    vocab = {str(token): tid for token, tid in hf_vocab.items()}

    # Ensure special tokens are mapped correctly at expected IDs
    for i, s in enumerate(SPECIAL_TOKENS):
        vocab[s] = i

    # Convert merges to [left, right] format
    merges = []
    seen = set()
    for m in hf_merges:
        if isinstance(m, (list, tuple)) and len(m) == 2:
            left, right = str(m[0]), str(m[1])
            key = f"{left}\0{right}"
            if key not in seen:
                seen.add(key)
                merges.append([left, right])

    config = {
        "vocabSize": VOCAB_SIZE,
        "minFrequency": MIN_FREQUENCY,
        "preTokenizer": "char",
        "specialTokens": SPECIAL_TOKENS,
    }

    return {"vocab": vocab, "merges": merges, "config": config}


def main():
    # Extract texts
    texts = extract_texts(DATASET_PATH)

    # Train BPE
    tokenizer = train_bpe(texts)

    # Convert to bpe.rs format
    output_data = convert_to_bpe_rs_format(tokenizer)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)

    print(f"\n💾 Saved to {OUTPUT_PATH}")
    print(f"   Vocab size: {len(output_data['vocab'])}")
    print(f"   Merges count: {len(output_data['merges'])}")

    # Check a few tokens for inspection
    special_ids = {SPECIAL_TOKENS[i]: i for i in range(len(SPECIAL_TOKENS))}
    real_vocab = {t: i for t, i in output_data['vocab'].items() if i >= len(SPECIAL_TOKENS)}
    print(f"   Real tokens (non-special): {len(real_vocab)}")

    # Show sample tokens
    sample_tokens = list(real_vocab.items())[:10]
    print(f"   Sample tokens: {sample_tokens}")


if __name__ == "__main__":
    main()
