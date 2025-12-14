import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import CanineTokenizer, CanineModel


# Load CANINE Model + Tokenizer
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU
else:
    device = torch.device("cpu")

print("Using device:", device)

tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
model = CanineModel.from_pretrained("google/canine-c").to(device)
model.eval()



# Batch Embedding Function
def embed_batch(urls, max_length=256):
    """Embed a batch of URLs using CANINE CLS token."""
    tokens = tokenizer(
        urls,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)

    # CLS vector = first token in sequence
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return cls_emb.cpu().numpy()



# Embed the Entire Dataset Split
def embed_all(urls, batch_size=64):
    """Embed all URLs by accumulating batch embeddings."""
    all_embs = []
    for i in tqdm(range(0, len(urls), batch_size), desc="Embedding"):
        batch = urls[i:i + batch_size]
        embs = embed_batch(batch)
        all_embs.append(embs)

    return np.vstack(all_embs)


# Process One Dataset Split (train/val/testA)
def process_split(csv_path, prefix):
    """
    Loads a CSV split (train/val/testA), embeds URLs,
    and saves .npy vectors + labels.
    """

    print(f"\n=== Processing {prefix} ===")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'url' and 'label' columns.")

    urls = df["url"].astype(str).tolist()
    labels = df["label"].astype(int).values

    print(f"{prefix}: Found {len(urls)} URLs")

    # Embed URLs
    X = embed_all(urls, batch_size=32)

    # Save outputs
    np.save(f"../processed_data/embedded_data/{prefix}_X.npy", X)
    np.save(f"../processed_data/embedded_data/{prefix}_y.npy", labels)

    print(f"Saved {prefix}_X.npy   shape={X.shape}")
    print(f"Saved {prefix}_y.npy   shape={labels.shape}")


# Run for ALL 3 SPLITS
process_split("../processed_data/basic_data/train.csv", "train")
process_split("../processed_data/basic_data/val.csv",   "val")
process_split("../processed_data/basic_data/testA.csv", "testA")

print("\nAll splits embedded and saved successfully.")
