import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data_path, save_path):
    # LOAD DATA
    df = pd.read_csv(data_path)
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    print("TOTAL:", len(df))
    print("Phish:", sum(df.label == 1))
    print("Benign:", sum(df.label == 0))

    # ---------------------------------------------------------
    # 1) GLOBAL SHUFFLE + SINGLE TRAIN/VAL/TEST SPLIT (NO LEAK)
    # ---------------------------------------------------------
    train_full, temp = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
    val_full, testA_full = train_test_split(temp, test_size=0.50, random_state=42)

    print("\nINITIAL SPLITS (before rebalancing):")
    print("Train:", len(train_full))
    print("Val:", len(val_full))
    print("TestA:", len(testA_full))

    # ---------------------------------------------------------
    # 2) TRAIN SPLIT: BALANCED 1:1
    # ---------------------------------------------------------
    train_phish = train_full[train_full.label == 1]
    train_benign = train_full[train_full.label == 0]

    n = min(len(train_phish), len(train_benign))  # max balanced amount

    train_df = pd.concat([
        train_phish.sample(n=n, random_state=42),
        train_benign.sample(n=n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nTRAIN (Balanced 1:1):", len(train_df))
    print("  Benign:", sum(train_df.label == 0))
    print("  Phish :", sum(train_df.label == 1))

    # ---------------------------------------------------------
    # 3) VALIDATION SPLIT: IMBALANCED 1:5
    # ---------------------------------------------------------
    val_phish = val_full[val_full.label == 1]
    val_benign = val_full[val_full.label == 0]

    n_val_phish = min(len(val_phish), 5000)  # cap, or use all available phishing
    benign_needed = min(len(val_benign), 5 * n_val_phish)

    val_df = pd.concat([
        val_phish.sample(n=n_val_phish, random_state=42),
        val_benign.sample(n=benign_needed, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nVAL (target 1:5 imbalance, adjusted to available data):", len(val_df))
    print("  Benign:", sum(val_df.label == 0))
    print("  Phish :", sum(val_df.label == 1))

    # ---------------------------------------------------------
    # 4) TEST A SPLIT: IMBALANCED 1:10 (FIXED!)
    # ---------------------------------------------------------
    testA_phish = testA_full[testA_full.label == 1]
    testA_benign = testA_full[testA_full.label == 0]

    n_test_phish = min(len(testA_phish), 5000)  # use up to 5k phishing
    benign_needed_testA = min(len(testA_benign), 10 * n_test_phish)

    testA_df = pd.concat([
        testA_phish.sample(n=n_test_phish, random_state=42),
        testA_benign.sample(n=benign_needed_testA, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nTEST A (target 1:10 imbalance, adjusted to available data):", len(testA_df))
    print("  Benign:", sum(testA_df.label == 0))
    print("  Phish :", sum(testA_df.label == 1))

    # ---------------------------------------------------------
    # 5) SAVE ALL SPLITS
    # ---------------------------------------------------------
    train_df.to_csv(f"{save_path}/train.csv", index=False)
    val_df.to_csv(f"{save_path}/val.csv", index=False)
    testA_df.to_csv(f"{save_path}/testA.csv", index=False)

    print("\nSaved train.csv, val.csv, testA.csv")


datasets = ["basic_data.csv", "tier2_data.csv"]

print("Basic data splits..........")
split_data("../basic_data.csv", "../processed_data/basic_data")

print("engineered_data splits........")
split_data("../tier2_data.csv", "../processed_data/engineered_data")
