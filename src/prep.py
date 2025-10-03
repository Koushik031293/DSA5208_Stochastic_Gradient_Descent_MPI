############################data cleaning##########################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hashlib
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_COLS = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "extra",
        "total_amount",
]

#set dtype_map to reduce memory usage
dtype_map = {
    "passenger_count": "Float32",
    "trip_distance":   "Float32",
    "extra":           "Float32",
    "total_amount":    "Float32",
    "RatecodeID":   "Int16",
    "PULocationID": "Int16",
    "DOLocationID": "Int16",
    "payment_type": "Int8",
}

#read data file, keep only required columns and set dtypes
df = pd.read_csv(
    os.path.join(DATA_DIR, "nytaxi2022.csv"),
    usecols=REQUIRED_COLS,
    dtype=dtype_map,
    na_values=["\\N", "", "NA"], 
    keep_default_na=True,    
    low_memory=False
)

#set datetime format
dt_fmt = '%m/%d/%Y %I:%M:%S %p'
df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"], format=dt_fmt, errors="coerce")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], format=dt_fmt, errors="coerce")

#drop rows with missing values
df = df.dropna()

#basic sanity check on passenger_count, trip_distance, total_amount. 
#remove passenger_count <= 0, trip_distance <= 0, total_amount <= 0
df = df[(df["passenger_count"] > 0) & (df["trip_distance"] > 0) & (df["total_amount"] > 0)]

#add duration column in minutes
df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
#move duration_min column to be after tpep_dropoff_datetime
cols = list(df.columns)
# find index of 'tpep_dropoff_datetime'
idx = cols.index("tpep_dropoff_datetime")
# move duration_min right after tpep_dropoff_datetime
cols.insert(idx+1, cols.pop(cols.index("duration_min")))
df = df[cols]

#drop rows with duration_min <= 0
df = df[df["duration_min"] > 0]

#given nyc's speed limit of 25mph to 65mph i.e. 40km/h to 105km/h, remove trips impossible trips 
df["duration_hour"] = df["duration_min"] / 60.0
df["speed_kmh"] = df["trip_distance"] / df["duration_hour"] #lets just assume trip distance is in km
cols = list(df.columns)
# find index of 'duration_min'
idx = cols.index("duration_min")
# move duration_hour right after duration_min
cols.insert(idx+1, cols.pop(cols.index("duration_hour")))
# because we just moved a column, recompute index of duration_min
idx = cols.index("duration_min")
# move speed_kmh right after duration_min (which will then be after duration_hour)
cols.insert(idx+2, cols.pop(cols.index("speed_kmh")))
df = df[cols]

#drop nonsense rows, nyc speed limit is 40-105kmh
df = df[(df["speed_kmh"] >= 1) & (df["speed_kmh"] <= 120)]

#drop rows with total amount <3, which is the minimum fare in nyc
df = df[df["total_amount"] >= 3]

#drop rows with duration_min <= 1
df = df[df["duration_min"] > 1]

#calculate fare_per_km
df["fare_per_km"] = df["total_amount"] / df["trip_distance"]
#calculate fare_per_min
df["fare_per_min"] = df["total_amount"] / df["duration_min"]

#remove ratecodeid that is not either 1,2,3,4,5,6,99 reference: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
df = df[df["RatecodeID"].isin([1,2,3,4,5,6,99])]

# bounds per RatecodeID: (km_low, km_high, min_low, min_high)
bounds = {
    1: (0.5, 20, 0.10, 10),
    2: (0.5, 10, 0.03, 10),
    3: (0.5, 12, 0.05, 12),
    4: (0.5, 20, 0.05, 12),
    5: (0.05, 30, 0.02, 20),
    6: (0.05, 20, 0.02, 12),
    99: (0.05, 30, 0.05, 20),
}

# default fallback if unseen code appears
default_bounds = (0.05, 30, 0.05, 20)

# vectorized mask builder
def make_mask(s_code, km, per_min):
    km_lo, km_hi, m_lo, m_hi = zip(*[
        bounds.get(code, default_bounds) for code in s_code.values
    ])
    km_lo = np.array(km_lo); km_hi = np.array(km_hi)
    m_lo  = np.array(m_lo);  m_hi  = np.array(m_hi)
    return (km >= km_lo) & (km <= km_hi) & (per_min >= m_lo) & (per_min <= m_hi)

mask = make_mask(df["RatecodeID"], df["fare_per_km"].to_numpy(), df["fare_per_min"].to_numpy())

df = df[mask]

#export cleaned data yayyyyyyy
# df.to_csv("../data/nytaxi2022_cleaned.csv", index=False)
df.to_csv(os.path.join(OUTPUT_DIR, "nytaxi2022_cleaned.csv"), index=False)

##########################split function##########################
#split into train and test set, 70% train, 30% test


TRAIN_RATIO = 0.7
# OUTPUT_DIR = "../data/"   # adjust path as needed

#split function
def assign_split(keys, train_ratio=TRAIN_RATIO):
    """
    Deterministic hash-based split into train/test.
    keys: pd.Series of string identifiers for each row
    """
    hash_vals = keys.apply(
        lambda s: int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) % 10000
    )
    return np.where(hash_vals < int(train_ratio * 10000), "train", "test")
#Build keys for hashing, use only stable and unique columns (no NaNs, already cleaned)
keys = (
    df["tpep_pickup_datetime"].astype("int64").astype(str) + "|" +
    df["tpep_dropoff_datetime"].astype("int64").astype(str) + "|" +
    df["PULocationID"].astype(str) + "|" +
    df["DOLocationID"].astype(str) + "|" +
    df["trip_distance"].round(4).astype(str) + "|" +
    df["total_amount"].round(2).astype(str)
)

# Apply split
df["split"] = assign_split(keys)

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

print(f"Train size: {len(train_df):,} rows")
print(f"Test size:  {len(test_df):,} rows")
print(f"Train ratio: {len(train_df) / len(df):.3f}")



# ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# save parquet
train_df.to_parquet(os.path.join(OUTPUT_DIR, "taxi_train.parquet"), index=False)
test_df.to_parquet(os.path.join(OUTPUT_DIR, "taxi_test.parquet"), index=False)

# Optional: also CSV (larger files, slower IO)
train_df.to_csv(os.path.join(OUTPUT_DIR, "taxi_train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "taxi_test.csv"), index=False)

print("Export complete! Files written to:", OUTPUT_DIR)
