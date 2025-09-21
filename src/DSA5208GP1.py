import argparse
import math
import time
import numpy as np
import pandas as pd
from mpi4py import MPI

# Data loading & sharding
#--------------------------------
def load_parquet(train_path, test_path, ycol):
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

    # Make sure datetime dtype is proper
    ensure_dt(df_train, "tpep_pickup_datetime")
    ensure_dt(df_train, "tpep_dropoff_datetime")
    ensure_dt(df_test,  "tpep_pickup_datetime")
    ensure_dt(df_test,  "tpep_dropoff_datetime")

    def add_time_feats(df):
        # duration in minutes (if not already present)
        if "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
            if "duration_min" not in df.columns:
                dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
                df["duration_min"] = dur.astype(np.float32)
                df.loc[df["duration_min"] < 0, "duration_min"] = np.nan  # guard bad rows

        # pickup hour/day-of-week cyclical encodings 
        if "tpep_pickup_datetime" in df.columns:
            pu = df["tpep_pickup_datetime"]
            pu_hour = pu.dt.hour.astype("float32")
            pu_dow  = pu.dt.dayofweek.astype("float32")  # Mon=0..Sun=6

            df["pu_hour_sin"] = np.sin(2*np.pi*pu_hour/24.0).astype("float32")
            df["pu_hour_cos"] = np.cos(2*np.pi*pu_hour/24.0).astype("float32")
            df["pu_dow_sin"]  = np.sin(2*np.pi*pu_dow/7.0).astype("float32")
            df["pu_dow_cos"]  = np.cos(2*np.pi*pu_dow/7.0).astype("float32")

        # Drop raw datetime columns (model needs numeric only)
        for c in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        return df

    df_train = add_time_feats(df_train.copy())
    df_test  = add_time_feats(df_test.copy())

    # Select numeric feature columns from TRAIN and align TEST to same set
    feat_cols = df_train.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()

    # Fill NaNs using TRAIN medians
    train_medians = df_train[feat_cols].median(numeric_only=True)

    X_train = df_train[feat_cols].fillna(train_medians).to_numpy(dtype=np.float32)
    y_train = df_train[ycol].fillna(df_train[ycol].median()).to_numpy(dtype=np.float32)

    # Align test columns and fill with TRAIN medians
    X_test  = df_test.reindex(columns=feat_cols).fillna(train_medians).to_numpy(dtype=np.float32)
    y_test  = df_test[ycol].fillna(df_train[ycol].median()).to_numpy(dtype=np.float32)

    print("Using features:", feat_cols)

    return X_train, y_train, X_test, y_test

##Split the dataset (X, y) into nearly-equal shards for distributed training.
def shard_data(X, y, rank, world):
    # Total number of samples in the dataset
    N = X.shape[0]
    #an array of indices 
    idx = np.arange(N)

    # Calculate how many samples each worker should get.
    # Formula: ceiling division (N / world), so distribution is nearly even.
    per = (N + world - 1) // world

    # Determine the slice of data that this worker (rank) is responsible for.
    start = rank * per
    end = min(N, (rank + 1) * per)

    # Edge case: if there are more workers than samples,
    if start >= N:
        # Return empty arrays (same type/slice for safety).
        return X[:0], y[:0]

    # Return the shard of data assigned to this worker.
    return X[start:end], y[start:end]

# activations 
def act_fn(z, kind):
    if kind == "relu":
        return np.maximum(0.0, z)
    elif kind == "tanh":
        return np.tanh(z)
    elif kind == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"Unknown activation '{kind}'")

#gradients of activations
def act_grad(a, z, kind):
    # a = activation(z); some grads use 'a' directly for efficiency
    if kind == "relu":
        return (z > 0).astype(z.dtype)
    elif kind == "tanh":
        return 1.0 - a**2
    elif kind == "sigmoid":
        return a * (1.0 - a)
    else:
        raise ValueError(f"Unknown activation '{kind}'")


# One-hidden-layer MLP (scalar output)
class OneHiddenNN:
    def __init__(self, n_in, n_hidden, activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        # Xavier/Glorot-ish init
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
        # X: (B, d)
        Z1 = X @ self.W1 + self.b1[None, :]      # (B, H)
        A1 = act_fn(Z1, self.activation)         # (B, H)
        Yhat = A1 @ self.W2 + self.b2[None, :]   # (B, 1)
        return Z1, A1, Yhat

    def loss_mse(self, Yhat, y):
        # y: (B,) or (B,1)
        y = y.reshape(-1, 1)
        # 1/(2B) * sum (yhat - y)^2 (matches project’s 1/(2N))
        diff = (Yhat - y)
        return 0.5 * float(np.mean(diff * diff))

    def backward(self, X, Z1, A1, Yhat, y):
        # Compute gradients for current mini-batch (averaged over batch)
        B = X.shape[0]
        y = y.reshape(-1, 1)
        diff = (Yhat - y)  # (B,1)

        # dLoss/dW2, dLoss/db2
        # d(0.5/B * sum diff^2)/dW2 = (1/B) * A1^T @ diff
        self.gW2[:] = (A1.T @ diff) / B
        self.gb2[:] = np.mean(diff, axis=0)

        # Backprop to hidden
        dA1 = diff @ self.W2.T                          # (B, H)
        dZ1 = dA1 * act_grad(A1, Z1, self.activation)   # (B, H)

        # dLoss/dW1, dLoss/db1
        self.gW1[:] = (X.T @ dZ1) / B                   # (d, H)
        self.gb1[:] = np.mean(dZ1, axis=0)              # (H,)

    def apply_gradients(self, lr):
        self.W1 -= lr * self.gW1
        self.b1 -= lr * self.gb1
        self.W2 -= lr * self.gW2
        self.b2 -= lr * self.gb2

    def predict(self, X, batch=4096):
        # Memory-friendly predictions
        out = []
        for i in range(0, X.shape[0], batch):
            _, _, yhat = self.forward(X[i:i+batch])
            out.append(yhat)
        return np.vstack(out).reshape(-1)

# -----------------------------
# MPI helpers
# -----------------------------
def mpi_average(value, comm):
    total = np.array([value], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, total, op=MPI.SUM)
    total /= comm.Get_size()
    return float(total[0])

def allreduce_inplace(arr, comm):
    """Allreduce (sum) into arr (float32/float64), then divide by world size to get mean."""
    tmp = np.array(arr, dtype=np.float64, copy=True)
    comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
    tmp /= comm.Get_size()
    arr[:] = tmp.astype(arr.dtype)



# RMSE (parallel)
def rmse_parallel(model, X, y, comm, batch=8192):
    # Each rank works on its own slice; results are aggregated
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

    # Reduce sums
    buf = np.array([local_se, local_cnt], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    tot_se, tot_cnt = buf
    return math.sqrt(tot_se / max(tot_cnt, 1))


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
    world = comm.Get_size()

    hist = []
    best_loss = float("inf")
    best_snap = None
    since_best = 0

    n_local = X_local.shape[0]
    if n_local == 0:

        pass

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # Shuffle local shard
        if shuffle and n_local > 0:
            order = rng.permutation(n_local)
            X_local = X_local[order]
            y_local = y_local[order]

        # Mini-batch loop on each rank, sampling from LOCAL shard
        if n_local == 0:
            # Still need to zero gradients and participate in allreduces per step count
            steps = 0
        else:
            steps = (n_local + batch_size - 1) // batch_size

        # We’ll track per-epoch average loss (properly averaged across ranks)
        epoch_loss_acc = 0.0
        epoch_count_acc = 0

        for b in range(steps if steps > 0 else 1):
            if n_local == 0:
                # Empty grads; still allreduce the zeros to keep in sync
                model.gW1.fill(0.0); model.gb1.fill(0.0)
                model.gW2.fill(0.0); model.gb2.fill(0.0)
                # Dummy local loss
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
        if improved:
            best_loss = epoch_loss
            since_best = 0
            # snapshot weights
            best_snap = (
                model.W1.copy(), model.b1.copy(),
                model.W2.copy(), model.b2.copy()
            )
        else:
            since_best += 1

        if rank == 0 and (ep == 1 or ep % log_every == 0 or ep == epochs):
            print(f"[Epoch {ep:4d}] loss={epoch_loss:.6f} (best {best_loss:.6f}) " f"steps={epoch_count_acc} time={time.time()-t0:.2f}s")

        if patience is not None and since_best >= patience:
            if rank == 0:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs).")
            break

    # Restore best snapshot
    if best_snap is not None:
        model.W1[:], model.b1[:], model.W2[:], model.b2[:] = best_snap

    return hist


# Main
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
    return p.parse_args()

def main(args=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Only parse CLI if nothing was passed in
    if args is None:
        args = parse_args()

    if rank == 0:
        print("Loading parquet…")
    Xtr, ytr, Xte, yte = load_parquet(args.train, args.test, args.ycol)

    # Broadcast shapes for sanity
    d = Xtr.shape[1]
    d_all = comm.bcast(d if rank == 0 else None, root=0)
    assert d_all == d, "Feature dimension mismatch across ranks."

    # Shard training data across ranks (each rank uses only its local shard)
    X_local, y_local = shard_data(Xtr, ytr, rank, comm.Get_size())
    if rank == 0:
        print(f"Train shape global: {Xtr.shape}, Test: {Xte.shape}")
    print(f"[Rank {rank}] local shard: {X_local.shape}")

    # Stage 3: define model
    model = OneHiddenNN(n_in=d, n_hidden=args.hidden, activation=args.act, seed=2025 + rank)

    # Stage 4: train with synchronous SGD
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

    # Evaluate RMSE (parallel over the full dataset)
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

if __name__ == "__main__":
    main()
