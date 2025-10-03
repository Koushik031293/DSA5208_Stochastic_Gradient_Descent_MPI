import os
import argparse
import numpy as np
import pandas as pd
import math ,json

REQ_COL = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "extra",
    "total_amount"
]

ALLOWED_RATECODES = {1,2,3,4,5,6,99}

# Bounds per RatecodeID: (fare_per_km_low, fare_per_km_high, fare_per_min_low, fare_per_min_high)
RATECODE_BOUNDS = {
    1: (0.5, 20.0, 0.10, 10.0),   # Standard
    2: (0.5, 10.0, 0.03, 10.0),   # JFK
    3: (0.5, 12.0, 0.05, 12.0),   # Newark
    4: (0.5, 20.0, 0.05, 12.0),   # Nassau/Westchester
    5: (0.05, 30.0, 0.02, 20.0),  # Negotiated
    6: (0.05, 20.0, 0.02, 12.0),  # Group rides
    99:(0.05, 30.0, 0.05, 20.0),  # Unknown
}

MI_TO_KM = 1.60934
# DATE_FMT = "%Y-%m-%d %H:%M:%S"
DATE_FMT = "%m/%d/%Y %I:%M:%S %p"

def _encode_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the two datetime columns to numeric encodings + duration feature."""
    df = df.copy()
    for c in ["tpep_pickup_datetime","tpep_dropoff_datetime"]:
        df[c] = pd.to_datetime(df[c], format=DATE_FMT, errors="coerce")
    # Duration in minutes
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    # Numeric encodings (minutes since epoch)
    df["pickup_min"]  = (df["tpep_pickup_datetime"].astype("int64") // 10**9) / 60.0
    df["dropoff_min"] = (df["tpep_dropoff_datetime"].astype("int64") // 10**9) / 60.0
    # Keep required numeric-friendly columns
    keep = [
        "passenger_count","trip_distance","RatecodeID","PULocationID","DOLocationID",
        "payment_type","extra","duration_min","pickup_min","dropoff_min","total_amount"
    ]
    return df[keep]

def apply_quality_rules(df_raw: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """
    Apply preprocessing rules before splitting.
    """
    df = df_raw.copy()

    # Required columns not null
    needed = list(REQ_COL) + [ycol]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    # Non-negativity checks
    for c in ["passenger_count","trip_distance","extra"]:
        if c in df.columns:
            df = df[df[c] >= 0]

    # Datetimes (fixed format for speed/consistency)
    for c in ["tpep_pickup_datetime","tpep_dropoff_datetime"]:
        df[c] = pd.to_datetime(df[c], format=DATE_FMT, errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime"])

    # Duration and minimum duration
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
    df = df[df["duration_min"] >= 1.0]

    # Speed sanity: 1 ≤ km/h ≤ 120
    dist_km = df["trip_distance"] * MI_TO_KM
    hours   = df["duration_min"] / 60.0
    speed_kmh = dist_km / hours.replace(0, np.nan)
    df = df[(speed_kmh >= 1.0) & (speed_kmh <= 120.0)]

    # Recompute after speed filter to keep arrays aligned
    df = df[df[ycol] >= 3.0]

    # Ratecode restriction
    df = df[df["RatecodeID"].isin(ALLOWED_RATECODES)]

    # --- compute per-km/per-min using the CURRENT df (recompute dist_km now) ---
    dist_km = df["trip_distance"] * MI_TO_KM   # recompute AFTER all filters above
    fare_col = ycol  # total_amount is the fare

    per_km  = np.where(dist_km > 0, df[fare_col] / dist_km, np.inf)
    per_min = df[fare_col] / df["duration_min"].replace(0, np.nan)

    def in_bounds(row):
        rc = int(row["RatecodeID"])
        bounds = RATECODE_BOUNDS.get(rc)
        if bounds is None:
            return False
        km_lo, km_hi, min_lo, min_hi = bounds
        return (row["_per_km"] >= km_lo) and (row["_per_km"] <= km_hi) and \
               (row["_per_min"] >= min_lo) and (row["_per_min"] <= min_hi)

    tmp = df.copy()
    # Sanity guard (optional during debugging)
    assert len(per_km) == len(tmp), (len(per_km), len(tmp))
    assert len(per_min) == len(tmp), (len(per_min), len(tmp))

    tmp["_per_km"]  = per_km
    tmp["_per_min"] = per_min
    tmp = tmp[tmp.apply(in_bounds, axis=1)].drop(columns=["_per_km","_per_min"])
    return tmp

def train_test_split(df: pd.DataFrame, seed: int = 42):
    if 'split' in df.columns:
        tr = df[df['split'].astype(str).str.lower().eq('train')].copy()
        te = df[df['split'].astype(str).str.lower().eq('test')].copy()
        return tr, te
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(df)) < 0.7
    return df[mask].copy(), df[~mask].copy()

def fit_normalizer(X: pd.DataFrame):
    mean = X.mean(axis=0).values
    std  = X.std(axis=0).replace(0, 1.0).values
    return mean, std

def apply_normalizer(X: pd.DataFrame, mean, std):
    return (X - mean) / std

def main():
    ap = argparse.ArgumentParser(description="NYC taxi preprocessing with rules + normalization + split")
    ap.add_argument("--csv", required=True, help="Path to nytaxi2022.csv")
    ap.add_argument("--outdir", default="data", help="Output directory (default: data)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for 70/30 split")
    ap.add_argument("--ycol", default="total_amount", help="Target column (default: total_amount)")
    ap.add_argument("--emit-combined", action="store_true", help="Also write one combined file with 'split'")
    ap.add_argument("--save-csv", action="store_true", help="Also save CSV copies")
    ap.add_argument("--scaler-npz", default=None, help="Optional path to save scaler as .npz (mean/std)")
    ap.add_argument("--meta-json", default=None, help="Optional path to save feature meta as .json")
    ap.add_argument("--compression", default="snappy", help="Parquet compression (snappy|gzip|brotli|none)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Read only needed columns with explicit dtypes; try fast pyarrow engine
    usecols = [
        "tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","trip_distance",
        "RatecodeID","PULocationID","DOLocationID","payment_type","extra",
        args.ycol, "total_amount"
    ]
    usecols = list(dict.fromkeys(usecols))  # dedupe

    dtypes = {
        "passenger_count":"Int64","trip_distance":"float64","RatecodeID":"Int64",
        "PULocationID":"Int64","DOLocationID":"Int64","payment_type":"Int64",
        "extra":"float64", args.ycol:"float64", "total_amount":"float64",
    }

    try:
        df_raw = pd.read_csv(args.csv, usecols=usecols, dtype=dtypes, engine="pyarrow")
    except Exception:
        df_raw = pd.read_csv(args.csv, usecols=usecols, dtype=dtypes, low_memory=False)

    # Ensure required columns exist
    missing = [c for c in REQ_COL if c not in df_raw.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if args.ycol not in df_raw.columns:
        raise ValueError(f"CSV missing target column '{args.ycol}'.")

    # Apply quality rules
    df_clean = apply_quality_rules(df_raw, args.ycol)

    # Encode datetimes + build feature matrix
    X = _encode_datetimes(df_clean)
    y = df_clean[args.ycol].astype(float).copy()

    # safety: drop any residual NaNs
    keep_mask = X.notna().all(axis=1) & y.notna()
    X = X[keep_mask].reset_index(drop=True)
    y = y[keep_mask].reset_index(drop=True)

    # Split
    Xtr, Xte = train_test_split(X, seed=args.seed)
    idx_tr = Xtr.index.to_numpy()
    idx_te = Xte.index.to_numpy()
    
    ytr = y.iloc[idx_tr].reset_index(drop=True)
    yte = y.iloc[idx_te].reset_index(drop=True)

    # Reset X indices to match y indices
    Xtr = Xtr.reset_index(drop=True)
    Xte = Xte.reset_index(drop=True)

    assert len(Xtr) == len(ytr), (len(Xtr), len(ytr))
    assert len(Xte) == len(yte), (len(Xte), len(yte))
    # Normalize (fit on train)
    mean, std = fit_normalizer(Xtr)
    Xtr_n = apply_normalizer(Xtr, mean, std)
    Xte_n = apply_normalizer(Xte, mean, std)

    # Save parquet
    Xtr_path_pq = os.path.join(args.outdir, "taxi_train.parquet")
    Xte_path_pq = os.path.join(args.outdir, "taxi_test.parquet")
    Xtr_n.assign(**{args.ycol: ytr.values}).to_parquet(Xtr_path_pq, index=False, compression=args.compression)
    Xte_n.assign(**{args.ycol: yte.values}).to_parquet(Xte_path_pq, index=False, compression=args.compression)

    # Optional CSV
    if args.save_csv:
        Xtr_n.assign(**{args.ycol: ytr.values}).to_csv(os.path.join(args.outdir, "taxi_train.csv"), index=False)
        Xte_n.assign(**{args.ycol: yte.values}).to_csv(os.path.join(args.outdir, "taxi_test.csv"), index=False)

    # # Optional combined
    # if args.emit_combined:
    #     combined = pd.concat([
    #         Xtr_n.assign(**{args.ycol: ytr.values}).assign(split="train"),
    #         Xte_n.assign(**{args.ycol: yte.values}).assign(split="test"),
    #     ], ignore_index=True)
    #     combined.to_parquet(os.path.join(args.outdir, "taxi_all.parquet"), index=False, compression=args.compression)
    #     if args.save_csv:
    #         combined.to_csv(os.path.join(args.outdir, "taxi_all.csv"), index=False)


    # Save scaler
    if args.scaler_npz:
        np.savez(args.scaler_npz, mean=mean, std=std)

    # Convert numpy / NA types into JSON-safe lists
    def _to_jsonable_list(arr):
        out = []
        for v in arr:
            if v is pd.NA or (isinstance(v, float) and math.isnan(v)):
                out.append(None)
            elif isinstance(v, (np.floating, np.integer)):
                out.append(v.item())
            else:
                out.append(float(v))
        return out

    meta = {
        "feature_names": [str(c) for c in X.columns],
        "mean": _to_jsonable_list(mean),
        "std": _to_jsonable_list(std),
        "seed": args.seed,
        "target": args.ycol,
        "rules_version": "v1-clean",
    }

    meta_path = args.meta_json if args.meta_json else os.path.join(args.outdir, "normalizer.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, allow_nan=False)

    print(f"Rows after cleaning: {len(df_clean):,}")
    print(f"✓ Wrote {Xtr_path_pq} and {Xte_path_pq}")
    if args.emit_combined:
        print("✓ Wrote combined dataset with 'split'")
    if args.save_csv:
        print("✓ Wrote CSV copies of train/test (and combined if requested)")
    print(f"✓ Saved scaler/meta to: {args.scaler_npz or '(skipped npz)'} ; {meta_path}")
    print(f"Features used: {list(X.columns)}")

if __name__ == "__main__":
    main()
