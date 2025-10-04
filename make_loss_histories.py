import os
import pandas as pd

# Root where your scaling runs are
SCALING_ROOT = "/Users/shreya-panchetty/Desktop/DSA5208/project/DSA5208_Stochastic_Gradient_Descent_MPI-dev/results/scaling"
# Where to collect new loss files
OUT_DIR = "/Users/shreya-panchetty/Desktop/DSA5208/results/loss_histories"

os.makedirs(OUT_DIR, exist_ok=True)

for run in os.scandir(SCALING_ROOT):
    if not run.is_dir():
        continue

    results_path = os.path.join(run.path, "results.csv")
    loss_path = os.path.join(run.path, "loss_curve.csv")

    if not (os.path.exists(results_path) and os.path.exists(loss_path)):
        continue

    try:
        # Read metadata from results.csv
        meta = pd.read_csv(results_path)
        if meta.empty:
            continue
        row = meta.iloc[0]
        act = str(row["activation"]).lower()
        batch = int(row["batch"])
        world = int(row["world"])
    except Exception as e:
        print(f"[WARN] Could not parse {results_path}: {e}")
        continue

    try:
        # Read the loss curve
        df_loss = pd.read_csv(loss_path)
    except Exception as e:
        print(f"[WARN] Could not read {loss_path}: {e}")
        continue

    # Build new filename
    out_name = f"loss_{act}_bs{batch}_w{world}.csv"
    out_file = os.path.join(OUT_DIR, out_name)

    df_loss.to_csv(out_file, index=False)
    print(f"[OK] Wrote {out_file}")
