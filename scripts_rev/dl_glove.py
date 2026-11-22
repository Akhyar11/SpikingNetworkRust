import os
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
print(snapshot_download('sentence-transformers/average_word_embeddings_glove.840B.300d',allow_patterns=['*.safetensors','*.json','*.md']))
