import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/mnist")
TRAIN_PARQUET = DATA_DIR / "train-00000-of-00001.parquet"
VALIDATION_SPLIT = 0.1


def main():
    df = pd.read_parquet(TRAIN_PARQUET)
    print(f"Loaded {len(df)} training samples")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(len(df_shuffled) * (1 - VALIDATION_SPLIT))
    train_df = df_shuffled[:split_idx]
    val_df = df_shuffled[split_idx:]

    train_df.to_parquet(DATA_DIR / "train_split.parquet", index=False)
    val_df.to_parquet(DATA_DIR / "val_split.parquet", index=False)

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")


if __name__ == "__main__":
    main()
