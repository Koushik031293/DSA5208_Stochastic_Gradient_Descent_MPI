import os
import pandas as pd

# ---- CONFIG: change these two paths if your layout differs ----
SCALING_ROOT = "/Users/shreya-panchetty/Desktop/DSA5208/project/DSA5208_Stochastic_Gradient_Descent_MPI-dev/results/scaling"
OUT_SUMMARY  = "/Users/shreya-panchetty/Desktop/DSA5208/results/summary.csv"
# ---------------------------------------------------------------

REQUIRED_COLS = {
    "timestamp", "host", "world", "activation", "hidden", "batch", "lr",
    # some runs call it epochs_run (we'll normalize to 'epochs')
    "train_rmse", "test_rmse", "train_time_s"
}

rows = []
for entry in os.scandir(SCALING_ROOT):
    if not entry.is_dir():
        continue

    run_dir = entry.path
    results_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(results_path):
        # skip folders without results.csv
        continue

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"[WARN] Failed to read {results_path}: {e}")
        continue

    if df.empty:
        print(f"[WARN] Empty results: {results_path}")
        continue

    # Normalize epoch column name
    if "epochs_run" in df.columns and "epochs" not in df.columns:
        df = df.rename(columns={"epochs_run": "epochs"})

    # Make sure we have all required columns (after normalization)
    have = set(df.columns)
    missing = REQUIRED_COLS - have
    # 'epochs' is optional but nice to have; REQUIRED_COLS doesn't include it
    if missing:
        print(f"[WARN] {results_path} missing columns {missing}. Keeping only available columns.")

    # Keep only the first row (your results.csv appears to contain a single-row summary)
    row = df.iloc[0].to_dict()

    # If epochs is missing, set None (script downstream can handle it)
    if "epochs" not in row:
        row["epochs"] = None

    # Ensure essential keys exist even if missing in file
    for k in ["timestamp","host","world","activation","hidden","batch","lr","train_rmse","test_rmse","train_time_s"]:
        row.setdefault(k, None)

    rows.append(row)

# Build the final summary DataFrame with a canonical column order
order = [
    "timestamp","host","activation","world","batch","hidden","lr",
    "epochs","train_rmse","test_rmse","train_time_s"
]
summary = pd.DataFrame(rows)
# Add any missing columns so we can save with a stable schema
for col in order:
    if col not in summary.columns:
        summary[col] = None
summary = summary[order]

# Clean types where possible
for c in ["world","batch","hidden","epochs"]:
    if c in summary.columns:
        summary[c] = pd.to_numeric(summary[c], errors="coerce").astype("Int64")
for c in ["lr","train_rmse","test_rmse","train_time_s"]:
    if c in summary.columns:
        summary[c] = pd.to_numeric(summary[c], errors="coerce")

# Sort (optional but handy)
summary = summary.sort_values(["activation","world","batch","timestamp"], ignore_index=True)

# Write it out
os.makedirs(os.path.dirname(OUT_SUMMARY), exist_ok=True)
summary.to_csv(OUT_SUMMARY, index=False)
print(f"[OK] Wrote {OUT_SUMMARY} with {len(summary)} rows.")