import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import CanineTokenizer, CanineModel

# # LOAD MODEL
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
# model = CanineModel.from_pretrained("google/canine-c").to(device)
# model.eval()
#
# # EMBEDDING FUNCTION
# def embed_batch(urls, max_length=256):
#     tokens = tokenizer(
#         urls,
#         padding=True,
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt"
#     ).to(device)
#
#     with torch.no_grad():
#         outputs = model(**tokens)
#
#     cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
#     return cls_emb.cpu().numpy()
#
#
# def embed_all(urls, batch_size=64):
#     all_embs = []
#     for i in tqdm(range(0, len(urls), batch_size)):
#         batch = urls[i:i+batch_size]
#         all_embs.append(embed_batch(batch))
#     return np.vstack(all_embs)
#
# df = pd.read_csv("../basic_data.csv")
# urls = df["url"].astype(str).tolist()
# labels = df["label"].values
#
# print("Embedding entire dataset...")
# X = embed_all(urls, batch_size=32)
#
# np.save("all_X.npy", X)
# np.save("all_y.npy", labels)
#
# print("Saved all_X.npy and all_y.npy")


import numpy as np
import pandas as pd

# Load embeddings
X = np.load("all_X.npy")
y = np.load("all_y.npy")

df = pd.DataFrame({"label": y})
df["idx"] = np.arange(len(df))    # keep original order

phish_idx  = df[df.label == 1].idx.values
benign_idx = df[df.label == 0].idx.values


# TRAIN (balanced 1:1)
np.random.seed(42)

train_phish  = phish_idx
train_benign = np.random.choice(benign_idx, size=len(train_phish), replace=False)

train_idx = np.concatenate([train_phish, train_benign])
np.random.shuffle(train_idx)


# VALIDATION 1:5
val_phish  = np.random.choice(phish_idx,  5000, replace=False)
val_benign = np.random.choice(benign_idx, 25000, replace=False)

val_idx = np.concatenate([val_phish, val_benign])
np.random.shuffle(val_idx)


# TEST A (1:10)
np.random.seed(100)

testA_phish  = np.random.choice(phish_idx,  5000, replace=False)
testA_benign = np.random.choice(benign_idx, 50000, replace=False)

testA_idx = np.concatenate([testA_phish, testA_benign])
np.random.shuffle(testA_idx)

# ---------------------------
# TEST B (1:20)
np.random.seed(200)

testB_phish  = np.random.choice(phish_idx,  5000, replace=False)
testB_benign = np.random.choice(benign_idx, 100000, replace=False)

testB_idx = np.concatenate([testB_phish, testB_benign])
np.random.shuffle(testB_idx)


# SAVE SPLITS
np.save("../processed_data/embedded_data/train_X.npy", X[train_idx])
np.save("../processed_data/embedded_data/train_y.npy", y[train_idx])

np.save("../processed_data/embedded_data/val_X.npy", X[val_idx])
np.save("../processed_data/embedded_data/val_y.npy", y[val_idx])

np.save("../processed_data/embedded_data/testA_X.npy", X[testA_idx])
np.save("../processed_data/embedded_data/testA_y.npy", y[testA_idx])

np.save("../processed_data/embedded_data/testB_X.npy", X[testB_idx])
np.save("../processed_data/embedded_data/testB_y.npy", y[testB_idx])

print("Saved train, val, testA, testB .npy files.")
