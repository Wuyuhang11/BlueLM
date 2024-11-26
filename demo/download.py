import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
from huggingface_hub import login, snapshot_download

login(token="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")
ds = load_dataset("greyfoss/doupo-dataset")