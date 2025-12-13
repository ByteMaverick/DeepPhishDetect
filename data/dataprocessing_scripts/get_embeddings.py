from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd




# Load data
df = pd.read_csv("../basic_data.csv")


# Load model
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Send model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_batch_embeddings(text_batch):
    """Compute CLS embeddings for a batch of URLs."""
    # Tokenize batch
    inputs = tokenizer(
        text_batch,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    # Move to CPU and convert to numpy
    return cls_embeddings.cpu().numpy()


# BATCHEd Processing
BATCH_SIZE = 64
urls = df["url"].tolist()

all_embeddings = []

for i in tqdm(range(0, len(urls), BATCH_SIZE), desc="Embedding URLs"):
    batch_urls = urls[i : i + BATCH_SIZE]
    batch_emb = get_batch_embeddings(batch_urls)
    all_embeddings.append(batch_emb)

# Stack all into final array
embeddings = np.vstack(all_embeddings)
np.save("../processed_data/embedded_data/embeddings.npy", embeddings)
print("Final embedding shape:", embeddings.shape)
