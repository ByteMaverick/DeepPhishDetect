import pandas as pd
import numpy as np

# Load the full dataset used for embedding generation
full_df = pd.read_csv("../basic_data.csv")
full_df["row_index"] = full_df.index

train_df = pd.read_csv("../processed_data/basic_data/train.csv")
val_df   = pd.read_csv("../processed_data/basic_data/val.csv")
testA_df = pd.read_csv("../processed_data/basic_data/testA.csv")
testB_df = pd.read_csv("../processed_data/basic_data/testB.csv")


train_df = train_df.merge(full_df[["u rl", "row_index"]], on="url", how="left")
val_df   = val_df.merge(full_df[["url", "row_index"]], on="url", how="left")
testA_df = testA_df.merge(full_df[["url", "row_index"]], on="url", how="left")
testB_df = testB_df.merge(full_df[["url", "row_index"]], on="url", how="left")


embeddings = np.load("../processed_data/embedded_data/embeddings.npy")
print(embeddings.shape)

train_emb = embeddings[train_df["row_index"].values]
val_emb   = embeddings[val_df["row_index"].values]
testA_emb = embeddings[testA_df["row_index"].values]
testB_emb = embeddings[testB_df["row_index"].values]

np.save("../processed_data/embedded_data/train_embeddings.npy", train_emb)
np.save("../processed_data/embedded_data/val_embeddings.npy",   val_emb)
np.save("../processed_data/embedded_data/testA_embeddings.npy", testA_emb)
np.save("../processed_data/embedded_data/testB_embeddings.npy", testB_emb)
