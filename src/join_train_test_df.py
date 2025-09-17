import pandas as pd
import os

DATA_DIR = "data"   # or "../data" depending on your path

# Load the two split files
train_df = pd.read_csv(os.path.join(DATA_DIR, "taxi_train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "taxi_test.csv"))

# Add split labels
train_df["split"] = "train"
test_df["split"]  = "test"

# Concatenate back together
df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Save combined file
df.to_csv(os.path.join(DATA_DIR, "nytaxi2022_split.csv"), index=False)

print("Combined dataset shape:", df.shape)
print(df["split"].value_counts())
