from datasets import load_dataset
try:
    ds = load_dataset("allenai/nllb", "eng_Latn-ind_Latn", split="train", streaming=True, trust_remote_code=True)
    for i, row in enumerate(ds):
        print(row)
        if i >= 1:
            break
except Exception as e:
    print(f"Error: {e}")
