import argparse
import math
import time
import numpy as np
import pandas as pd
from mpi4py import MPI
import os, csv, json
from datetime import datetime
import socket
import matplotlib.pyplot as plt

# =============================================================
# Data loading & feature engineering
# =============================================================

def load_parquet(train_path, test_path, ycol):
    """Load train/test parquet files, engineer time features, sanitize, and standardize.
    Returns: X_train, y_train, X_test, y_test (all np.float32)
    """
    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)

    # Ensure target numeric
    df_train[ycol] = pd.to_numeric(df_train[ycol], errors="coerce")
    df_test[ycol]  = pd.to_numeric(df_test[ycol],  errors="coerce")

    # Drop non-feature split-like columns if present
    drop_if_present = ["split", "dataset", "set", "part", "fold"]
    for c in drop_if_present:
        if c in df_train.columns: df_train = df_train.drop(columns=c)
        if c in df_test.columns:  df_test  = df_test.drop(columns=c)

    def ensure_dt(df, col):
        if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure datetime dtypes
    for col in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
        ensure_dt(df_train, col)
        ensure_dt(df_test,  col)

    def add_time_feats(df):
        # Duration minutes
        if {"tpep_pickup_datetime","tpep_dropoff_datetime"}.issubset(df.columns):
            if "duration_min" not in df.columns:
                dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
                df["duration_min"] = dur.astype(np.float32)
                df.loc[df["duration_min"] < 0, "duration_min"] = np.nan

        # Cyclical encodings for pickup time
        if "tpep_pickup_datetime" in df.columns:
            pu = df["tpep_pickup_datetime"]
            pu_hour = pu.dt.hour.astype("float32")
            pu_dow  = pu.dt.dayofweek.astype("float32")
            df["pu_hour_sin"] = np.sin(2*np.pi*pu_hour/24.0).astype("float32")
            df["pu_hour_cos"] = np.cos(2*np.pi*pu_hour/24.0).astype("float32")
            df["pu_dow_sin"]  = np.sin(2*np.pi*pu_dow/7.0).astype("float32")
            df["pu_dow_cos"]  = np.cos(2*np.pi*pu_dow/7.0).astype("float32")

        # Drop raw datetime cols to keep only numeric features
        for c in ("tpep_pickup_datetime", "tpep_dropoff_datetime"):
            if c in df.columns:
                df.drop(columns=c, inplace=True)
        return df

    df_train = add_time_feats(df_train.copy())
    df_test  = add_time_feats(df_test.copy())

    # Build numeric feature set from TRAIN only
    num_train = df_train.drop(columns=[ycol]).select_dtypes(include=[np.number]).copy()

    # Drop columns that are all-NaN in TRAIN (their medians would be NaN)
    all_nan_cols = [c for c in num_train.columns if num_train[c].isna().all()]
    if all_nan_cols:
        print("Dropping all-NaN columns (train):", all_nan_cols)
        num_train.drop(columns=all_nan_cols, inplace=True)

    feat_cols = num_train.columns.tolist()

    # Medians from TRAIN only
    train_medians = num_train.median(numeric_only=True)

    def sanitize(a):
        """Replace NaN/inf with finite values and cast to float32."""
        return np.nan_to_num(a, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)

    # Construct arrays
    X_train = sanitize(df_train[feat_cols].fillna(train_medians).to_numpy(dtype=np.float32))
    y_train = sanitize(df_train[ycol].fillna(df_train[ycol].median()).to_numpy(dtype=np.float32))

    X_test  = sanitize(df_test.reindex(columns=feat_cols).fillna(train_medians).to_numpy(dtype=np.float32))
    y_test  = sanitize(df_test[ycol].fillna(df_train[ycol].median()).to_numpy(dtype=np.float32))

    # Standardize features using TRAIN stats
    mu  = np.nanmean(X_train, axis=0).astype(np.float32)
    std = np.nanstd (X_train, axis=0).astype(np.float32)
    std[std == 0.0] = 1.0
    X_train = ((X_train - mu) / std).astype(np.float32)
    X_test  = ((X_test  - mu) / std).astype(np.float32)

    # Final sanitize after scaling
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1e6, neginf=-1e6)

    print("Using features:", feat_cols)
    return X_train, y_train, X_test, y_test


# =============================================================
# Sharding
# =============================================================

def shard_data(X, y, rank, world):
    N = X.shape[0]
    per = (N + world - 1) // world
    start = rank * per
    end = min(N, (rank + 1) * per)
    if start >= N:
        return X[:0], y[:0]
    return X[start:end], y[start:end]


# =============================================================
# Model: One-hidden-layer MLP (scalar output)
# =============================================================

def act_fn(z, kind):
    if kind == "relu":
        return np.maximum(0.0, z)
    if kind == "tanh":
        return np.tanh(z)
    if kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown activation '{kind}'")


def act_grad(a, z, kind):
    if kind == "relu":
        return (z > 0).astype(z.dtype)
    if kind == "tanh":
        return 1.0 - a**2
    if kind == "sigmoid":
        return a * (1.0 - a)
    raise ValueError(f"Unknown activation '{kind}'")


class OneHiddenNN:
    def __init__(self, n_in, n_hidden, activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        limit1 = math.sqrt(6.0 / (n_in + n_hidden))
        self.W1 = rng.uniform(-limit1, limit1, size=(n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros((n_hidden,), dtype=np.float32)
        limit2 = math.sqrt(6.0 / (n_hidden + 1))
        self.W2 = rng.uniform(-limit2, limit2, size=(n_hidden, 1)).astype(np.float32)
        self.b2 = np.zeros((1,), dtype=np.float32)
        self.activation = activation
        # Gradient buffers
        self.gW1 = np.zeros_like(self.W1)
        self.gb1 = np.zeros_like(self.b1)
        self.gW2 = np.zeros_like(self.W2)
        self.gb2 = np.zeros_like(self.b2)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1[None, :]
        A1 = act_fn(Z1, self.activation)
        Yhat = A1 @ self.W2 + self.b2[None, :]
        return Z1, A1, Yhat

    def loss_mse(self, Yhat, y):
        y = y.reshape(-1, 1)
        diff = (Yhat - y)
        return 0.5 * float(np.mean(diff * diff))

    def backward(self, X, Z1, A1, Yhat, y):
        B = X.shape[0]
        y = y.reshape(-1, 1)
        diff = (Yhat - y)
        self.gW2[:] = (A1.T @ diff) / B
        self.gb2[:] = np.mean(diff, axis=0)
        dA1 = diff @ self.W2.T
        dZ1 = dA1 * act_grad(A1, Z1, self.activation)
        self.gW1[:] = (X.T @ dZ1) / B
        self.gb1[:] = np.mean(dZ1, axis=0)

    def apply_gradients(self, lr):
        self.W1 -= lr * self.gW1
        self.b1 -= lr * self.gb1
        self.W2 -= lr * self.gW2
        self.b2 -= lr * self.gb2

    def predict(self, X, batch=4096):
        out = []
        for i in range(0, X.shape[0], batch):
            _, _, yhat = self.forward(X[i:i+batch])
            out.append(yhat)
        return np.vstack(out).reshape(-1)


# =============================================================
# MPI helpers & metrics
# =============================================================

def mpi_average(value, comm):
    total = np.array([value], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, total, op=MPI.SUM)
    total /= comm.Get_size()
    return float(total[0])


def allreduce_inplace(arr, comm):
    tmp = np.array(arr, dtype=np.float64, copy=True)
    comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
    tmp /= comm.Get_size()
    arr[:] = tmp.astype(arr.dtype)


def rmse_parallel(model, X, y, comm, batch=8192):
    rank = comm.Get_rank()
    world = comm.Get_size()
    N = X.shape[0]
    per = (N + world - 1) // world
    s = rank * per
    e = min(N, (rank + 1) * per)
    if s >= N:
        local_se = 0.0
        local_cnt = 0
    else:
        yhat = model.predict(X[s:e], batch=batch)
        diff = yhat - y[s:e]
        local_se = float(np.dot(diff, diff))
        local_cnt = int(diff.shape[0])
    buf = np.array([local_se, local_cnt], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    tot_se, tot_cnt = buf
    return math.sqrt(tot_se / max(tot_cnt, 1))


# =============================================================
# Training loop (synchronous SGD)
# =============================================================

def train_sgd_mpi(
    model,
    X_local, y_local,
    lr=1e-3,
    batch_size=512,
    epochs=50,
    shuffle=True,
    patience=10,
    comm=MPI.COMM_WORLD,
    log_every=10,
):
    rng = np.random.default_rng(123)
    rank = comm.Get_rank()

    hist = []
    best_loss = float("inf")
    best_snap = None
    since_best = 0

    n_local = X_local.shape[0]

    for ep in range(1, epochs + 1):
        t0 = time.time()
        if shuffle and n_local > 0:
            order = rng.permutation(n_local)
            X_local = X_local[order]
            y_local = y_local[order]

        steps = (n_local + batch_size - 1) // batch_size if n_local > 0 else 0
        epoch_loss_acc = 0.0
        epoch_count_acc = 0

        for b in range(steps if steps > 0 else 1):
            if n_local == 0:
                model.gW1.fill(0.0); model.gb1.fill(0.0)
                model.gW2.fill(0.0); model.gb2.fill(0.0)
                local_loss = 0.0
            else:
                s = b * batch_size
                e = min(n_local, s + batch_size)
                Xb = X_local[s:e]
                yb = y_local[s:e]

                Z1, A1, Yhat = model.forward(Xb)
                local_loss = model.loss_mse(Yhat, yb)
                model.backward(Xb, Z1, A1, Yhat, yb)

            # Average gradients across ranks (Allreduce + divide by world)
            allreduce_inplace(model.gW1, comm)
            allreduce_inplace(model.gb1, comm)
            allreduce_inplace(model.gW2, comm)
            allreduce_inplace(model.gb2, comm)

            # Gradient clipping + NaN guards (stability)
            for g in (model.gW1, model.gb1, model.gW2, model.gb2):
                np.clip(g, -1e2, 1e2, out=g)
            for g in (model.gW1, model.gb1, model.gW2, model.gb2):
                if not np.isfinite(g).all():
                    np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

            # Synchronized parameter update
            model.apply_gradients(lr)

            # Average the mini-batch loss across ranks for logging
            avg_batch_loss = mpi_average(local_loss, comm)
            epoch_loss_acc += avg_batch_loss
            epoch_count_acc += 1

        # Per-epoch average batch-loss across all steps
        epoch_loss = epoch_loss_acc / max(epoch_count_acc, 1)
        hist.append(epoch_loss)

        # Early stopping
        improved = epoch_loss < best_loss - 1e-8
        since_best = 0 if improved else since_best + 1
        if improved:
            best_loss = epoch_loss
            best_snap = (
                model.W1.copy(), model.b1.copy(),
                model.W2.copy(), model.b2.copy()
            )

        if rank == 0 and (ep == 1 or ep % log_every == 0 or ep == epochs):
            print(f"[Epoch {ep:4d}] loss={epoch_loss:.6f} (best {best_loss:.6f}) steps={epoch_count_acc} time={time.time()-t0:.2f}s")

        if patience is not None and since_best >= patience:
            if rank == 0:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs).")
            break

    # Restore best snapshot
    if best_snap is not None:
        model.W1[:], model.b1[:], model.W2[:], model.b2[:] = best_snap

    return hist


# =============================================================
# Saving helpers
# =============================================================

def save_results(args, hist, tr_rmse, te_rmse, train_time, model, Xte, yte):
    outdir = getattr(args, "outdir", "results/exp1")
    os.makedirs(outdir, exist_ok=True)

    # ---------------- 1) Summary row ----------------
    results = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "host": socket.gethostname(),
        "world": MPI.COMM_WORLD.Get_size(),
        "activation": args.act,
        "hidden": args.hidden,
        "batch": args.batch,
        "lr": args.lr,
        "epochs_run": len(hist),
        "train_rmse": float(tr_rmse),
        "test_rmse": float(te_rmse),
        "train_time_s": float(train_time),
    }
    outcsv = os.path.join(outdir, "results.csv")
    exists = os.path.isfile(outcsv)
    with open(outcsv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results.keys())
        if not exists: w.writeheader()
        w.writerow(results)

    # ---------------- 2) Loss history (optional) ----------------
    if getattr(args, "save_history", False):
        with open(os.path.join(outdir, "loss_curve.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","avg_batch_loss"])
            for i,v in enumerate(hist,1): w.writerow([i,float(v)])

    # ---------------- 3) Predictions + residuals ----------------
    yhat = model.predict(Xte, batch=8192)
    with open(os.path.join(outdir,"predictions_test.csv"),"w",newline="") as f:
        w = csv.writer(f); w.writerow(["y_true","y_pred"])
        for yt, yp in zip(yte, yhat): w.writerow([float(yt), float(yp)])

    # ---------------- 4) Plots (optional) ----------------
    if getattr(args, "plot_history", False) and plt is not None:
        # Loss curve
        plt.figure()
        plt.plot(range(1, len(hist)+1), hist)
        plt.xlabel("Epoch"); plt.ylabel("Avg batch loss"); plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=120)
        plt.close()

        # Residual histogram
        resid = (yhat - yte).astype(float)
        plt.figure()
        plt.hist(resid, bins=50)
        plt.xlabel("Residual (y_pred - y_true)"); plt.ylabel("Count"); plt.title("Residuals (test)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "residuals_hist.png"), dpi=120)
        plt.close()

        # Residual vs predicted (great to eyeball bias/variance)
        plt.figure()
        plt.scatter(yhat, resid, s=3, alpha=0.4)
        plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title("Residual vs Predicted (test)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "residuals_vs_pred.png"), dpi=120)
        plt.close()

    # ---------------- 5) Config snapshot ----------------
    with open(os.path.join(outdir,"run_config.json"),"w") as f:
        json.dump({k: getattr(args, k) for k in vars(args)}, f, indent=2)

    # ---------------- 6) Scalability log (append per run) ----------------
    # You can compute speedup later with your 1-rank baseline.
    scal_path = os.path.join(outdir, "scaling_runs.csv")
    scal_fields = ["timestamp","host","world","hidden","batch","lr","epochs_run","train_time_s","train_rmse","test_rmse"]
    scal_row = {
        "timestamp": results["timestamp"],
        "host": results["host"],
        "world": results["world"],
        "hidden": results["hidden"],
        "batch": results["batch"],
        "lr": results["lr"],
        "epochs_run": results["epochs_run"],
        "train_time_s": results["train_time_s"],
        "train_rmse": results["train_rmse"],
        "test_rmse": results["test_rmse"],
    }
    scal_exists = os.path.isfile(scal_path)
    with open(scal_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scal_fields)
        if not scal_exists: w.writeheader()
        w.writerow(scal_row)

    print(f"✓ Results saved in {outdir}")

# =============================================================
# CLI + entrypoint (standalone usage also supported)
# =============================================================

def parse_args():
    p = argparse.ArgumentParser(description="One-hidden-layer NN with synchronous SGD (MPI).")
    p.add_argument("--train", required=True, help="Path to preprocessed TRAIN parquet")
    p.add_argument("--test", required=True, help="Path to preprocessed TEST parquet")
    p.add_argument("--ycol", required=True, help="Target column name (e.g., total_amount)")
    p.add_argument("--hidden", type=int, default=64, help="Hidden units")
    p.add_argument("--act", choices=["relu", "tanh", "sigmoid"], default="relu", help="Activation")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--batch", type=int, default=512, help="Mini-batch size")
    p.add_argument("--epochs", type=int, default=200, help="Max epochs")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    p.add_argument("--outdir", default="results/exp1", help="Directory to store artifacts")
    p.add_argument("--save-history", action="store_true", help="Save loss_curve.csv")
    p.add_argument("--plot-history", action="store_true", help="Save PNG plots (loss & residual diagnostics)")
    return p.parse_args()


def main(args=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args is None:
        args = parse_args()

    if rank == 0:
        print("Loading parquet…")
    Xtr, ytr, Xte, yte = load_parquet(args.train, args.test, args.ycol)

    # Finite-data sanity logs
    if rank == 0:
        def stats(tag, X, y):
            print(f"[{tag}] X shape={X.shape} | nanX={np.isnan(X).sum()} infX={np.isinf(X).sum()} | "
                  f"nanY={np.isnan(y).sum()} infY={np.isinf(y).sum()}")
        stats("TRAIN", Xtr, ytr)
        stats("TEST",  Xte, yte)

    # Broadcast shapes for sanity
    d = Xtr.shape[1]
    d_all = comm.bcast(d if rank == 0 else None, root=0)
    assert d_all == d, "Feature dimension mismatch across ranks."

    # Shard training data across ranks
    X_local, y_local = shard_data(Xtr, ytr, rank, comm.Get_size())
    if rank == 0:
        print(f"Train shape global: {Xtr.shape}, Test: {Xte.shape}")
    print(f"[Rank {rank}] local shard: {X_local.shape}")

    # Define model
    model = OneHiddenNN(n_in=d, n_hidden=args.hidden, activation=args.act, seed=2025 + rank)

    # Train
    t0 = time.time()
    hist = train_sgd_mpi(
        model, X_local, y_local,
        lr=args.lr,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        comm=comm,
        log_every=5,
    )
    train_time = time.time() - t0

    # Evaluate (parallel RMSE over full sets)
    tr_rmse = rmse_parallel(model, Xtr, ytr, comm)
    te_rmse = rmse_parallel(model, Xte, yte, comm)

    if rank == 0:
        print("\n=== Results ===")
        print(f"Activation     : {args.act}")
        print(f"Hidden units   : {args.hidden}")
        print(f"Batch size     : {args.batch}")
        print(f"Learning rate  : {args.lr}")
        print(f"Epochs run     : {len(hist)}")
        print(f"Train RMSE     : {tr_rmse:.6f}")
        print(f"Test  RMSE     : {te_rmse:.6f}")
        print(f"Train time (s) : {train_time:.2f}")
        save_results(args, hist, tr_rmse, te_rmse, train_time, model, Xte, yte)


if __name__ == "__main__":
    main()
