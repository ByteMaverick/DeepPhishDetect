import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


datasets =["basic_data.csv","tier2_data.csv"]


def split_data(data_path,save_path):
        # LOAD DATA
        df = pd.read_csv(data_path)

        # Clean any whitespace
        df["url"] = df["url"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

        # Separate phishing and benign sets
        phish_df  = df[df["label"] == 1].reset_index(drop=True)
        benign_df = df[df["label"] == 0].reset_index(drop=True)

        print("Phishing URLs:", len(phish_df))
        print("Benign URLs:", len(benign_df))


        # SPLIT 1: TRAIN (balanced 1:1)
        # Use all phishing data
        phish_train = phish_df.copy()

        # Random sample equal number of benign
        benign_train = benign_df.sample(n=len(phish_train), random_state=42)

        train_df = pd.concat([phish_train, benign_train]).sample(frac=1, random_state=42)
        train_df = train_df.reset_index(drop=True)

        print("TRAIN SIZE:", len(train_df),
              " | Benign:", sum(train_df.label==0),
              " | Phish:", sum(train_df.label==1))


        # SPLIT 2: VALIDATION (1:5 imbalance)
        val_phish = phish_df.sample(n=5000, random_state=42)
        val_benign = benign_df.sample(n=25000, random_state=42)

        val_df = pd.concat([val_phish, val_benign]).sample(frac=1, random_state=42)
        val_df = val_df.reset_index(drop=True)

        print("VAL SIZE:", len(val_df),
              " | Benign:", sum(val_df.label==0),
              " | Phish:", sum(val_df.label==1))



        # SPLIT 3: TEST SET A (1:10 imbalance)
        testA_phish = phish_df.sample(n=5000, random_state=100)
        testA_benign = benign_df.sample(n=50000, random_state=100)

        testA_df = pd.concat([testA_phish, testA_benign]).sample(frac=1, random_state=100)
        testA_df = testA_df.reset_index(drop=True)

        print("TEST A SIZE:", len(testA_df),
              " | Benign:", sum(testA_df.label==0),

              " | Phish:", sum(testA_df.label==1))


        # SPLIT 4: TEST SET B (1:20 imbalance)
        testB_phish = phish_df.sample(n=5000, random_state=200)
        testB_benign = benign_df.sample(n=100000, random_state=200)

        testB_df = pd.concat([testB_phish, testB_benign]).sample(frac=1, random_state=200)
        testB_df = testB_df.reset_index(drop=True)

        print("TEST B SIZE:", len(testB_df),
              " | Benign:", sum(testB_df.label==0),
              " | Phish:", sum(testB_df.label==1))



        # SAVE ALL SPLITS
        train_df.to_csv(f"{save_path}/train.csv", index=False)
        val_df.to_csv(f"{save_path}/val.csv", index=False)
        testA_df.to_csv(f"{save_path}/testA.csv", index=False)
        testB_df.to_csv(f"{save_path}/testB.csv", index=False)


        print("\nSaved: train.csv, val.csv, testA.csv, testB.csv")



print("Basic data splits..........")
split_data("../basic_data.csv","../processed_data/basic_data")

print("engineered_data splits........")
split_data("../tier2_data.csv","../processed_data/engineered_data")

