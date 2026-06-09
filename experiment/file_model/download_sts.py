import json
from datasets import load_dataset
import os

datasets_to_download = [
    "mteb/sts12-sts",
    "mteb/sts13-sts",
    "mteb/sts14-sts",
    "mteb/sts15-sts",
    "mteb/sts16-sts",
    "mteb/sickr-sts"
]

out_dir = "/home/akhyar/Dokumen/Code/NODE_JS/SpikingNetworkRust/experiment/file_model"

for ds_name in datasets_to_download:
    print(f"Downloading {ds_name}...")
    try:
        ds = load_dataset(ds_name, split="test")
        formatted_data = []
        for row in ds:
            # HuggingFace datasets for STS usually have 'sentence1', 'sentence2', and 'score'
            score = row['score']
            # If the score is 0-5, scale to 0-1. But let's check max score.
            # actually usually MTEB STS scores are already 0-5. Let's scale it.
            # For SICK-R it's also 1-5. Let's dynamically scale based on max possible if needed,
            # or just assume score is 0-5 (or 1-5 for sickr)
            if score > 1.0:
                score = score / 5.0
            
            formatted_data.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "score": score
            })
        
        save_name = ds_name.replace("/", "_") + ".json"
        save_path = os.path.join(out_dir, save_name)
        with open(save_path, "w") as f:
            json.dump(formatted_data, f, indent=2)
        print(f"Saved {len(formatted_data)} pairs to {save_path}")
    except Exception as e:
        print(f"Failed to download {ds_name}: {e}")
