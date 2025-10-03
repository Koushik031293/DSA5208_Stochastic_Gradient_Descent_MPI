import os, math, time, csv, json, socket
from datetime import datetime
from xml.parsers.expat import model

import numpy as np
import pandas as pd
from mpi4py import MPI
comm = MPI.COMM_WORLD#new
rank = comm.Get_rank()#new
size = comm.Get_size()#new
def log(msg): print(f"[Rank {rank}] {msg}", flush=True)#new

# Hardening knobs to avoid native-thread crashes on macOS + MPI
os.environ.setdefault("ARROW_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

# Optional: disable interactive backends for remote MPI ranks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Feature engineering helpers
# ------------------------------------------------------------

def _ensure_dt(df: pd.DataFrame, col: str):
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")

def _add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    if {"tpep_pickup_datetime","tpep_dropoff_datetime"}.issubset(df.columns):
        dur = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60.0
        df["duration_min"] = dur.astype("float32")
        df.loc[df["duration_min"] < 0, "duration_min"] = np.nan

    if "tpep_pickup_datetime" in df.columns:
        pu = df["tpep_pickup_datetime"]
        df["pu_hour_sin"] = np.sin(2*np.pi*pu.dt.hour/24.0).astype("float32")
        df["pu_hour_cos"] = np.cos(2*np.pi*pu.dt.hour/24.0).astype("float32")
        df["pu_dow_sin"]  = np.sin(2*np.pi*pu.dt.dayofweek/7.0).astype("float32")
        df["pu_dow_cos"]  = np.cos(2*np.pi*pu.dt.dayofweek/7.0).astype("float32")

    df.drop(columns=["tpep_pickup_datetime","tpep_dropoff_datetime"], errors="ignore", inplace=True)
    return df

def _numeric_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    drop_if_present = ["split", "dataset", "set", "part", "fold"]
    df = df.drop(columns=[c for c in drop_if_present if c in df.columns], errors="ignore")
    if target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df
    return X.select_dtypes(include=[np.number]).columns.tolist()

def _sanitize(a: np.ndarray, cap=1e9) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=cap, neginf=-cap).astype("float32", copy=False)

# ------------------------------------------------------------
# Sharded Parquet loader
# ------------------------------------------------------------

def _rowgroup_ranges(pf, start_row: int, end_row: int):
    """Map a [start_row, end_row) slice to Parquet row-group indices + intra-group offsets."""
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    offsets = np.cumsum([0] + rg_sizes)
    # row groups covering our slice
    rg_start = max(i for i, off in enumerate(offsets) if off <= start_row)
    # include the last group that still starts before end_row
    rg_end = min(i for i, off in enumerate(offsets) if off < end_row)
    return rg_start, rg_end, offsets, rg_sizes

def load_parquet_sharded(train_path: str, test_path: str, ycol: str, comm: MPI.Comm):
    """
    Train: each rank reads only its shard (by rows mapped onto row groups).
    Test : rank 0 reads once and broadcasts to all ranks.
    Returns: (Xtr_local, ytr_local, Xte, yte, feat_cols)
    """
    import pyarrow.parquet as pq
    rank, world = comm.Get_rank(), comm.Get_size()

    # ---------- TRAIN ----------
    pf = pq.ParquetFile(train_path)
    nrows = pf.metadata.num_rows
    rows_per_rank = (nrows + world - 1) // world
    start = rank * rows_per_rank
    end   = min(nrows, (rank + 1) * rows_per_rank)

    if start >= end:
        df_train = pd.DataFrame()
    else:
        rg_start, rg_end, offsets, rg_sizes = _rowgroup_ranges(pf, start, end)
        # Read the covering row groups, then slice to exact rows
        tbl = pf.read_row_groups(list(range(rg_start, rg_end + 1)))
        df_train = tbl.to_pandas()
        head = start - offsets[rg_start]
        tail = head + (end - start)
        df_train = df_train.iloc[head:tail]

    # Target to float32
    if ycol in df_train.columns:
        df_train[ycol] = pd.to_numeric(df_train[ycol], errors="coerce").astype("float32")

    # Time features
    for col in ("tpep_pickup_datetime","tpep_dropoff_datetime"):
        _ensure_dt(df_train, col)
    df_train = _add_time_feats(df_train)

    # ---------- TEST ----------
    if rank == 0:
        df_test = pd.read_parquet(test_path, engine="pyarrow")
        df_test[ycol] = pd.to_numeric(df_test[ycol], errors="coerce").astype("float32")
        for col in ("tpep_pickup_datetime","tpep_dropoff_datetime"):
            _ensure_dt(df_test, col)
        df_test = _add_time_feats(df_test)
    else:
        df_test = None
    df_test = comm.bcast(df_test, root=0)

    # ---------- Feature columns: intersection across ranks ----------
    local_cols = set(_numeric_feature_columns(df_train, ycol))
    all_cols = comm.allgather(local_cols)
    feat_cols = sorted(set.intersection(*all_cols)) if all_cols else []

    if not feat_cols:  # Fallback
        feat_cols = sorted(local_cols)

    # ---------- Build arrays ----------
    Xtr = df_train.reindex(columns=feat_cols).to_numpy(dtype="float32", copy=False)
    ytr = df_train[ycol].to_numpy(dtype="float32", copy=False) if ycol in df_train else np.zeros((0,), dtype="float32")
    Xte = df_test.reindex(columns=feat_cols).to_numpy(dtype="float32", copy=False)
    yte = df_test[ycol].to_numpy(dtype="float32", copy=False)

    Xtr = _sanitize(Xtr); Xte = _sanitize(Xte)
    ytr = _sanitize(ytr); yte = _sanitize(yte)

    # ---------- Global mean/std from TRAIN shards ----------
    # sum, sumsq, count
    sum_local   = Xtr.sum(axis=0, dtype=np.float64) if Xtr.size else np.zeros((len(feat_cols),), dtype=np.float64)
    sumsq_local = (Xtr.astype(np.float64)**2).sum(axis=0) if Xtr.size else np.zeros((len(feat_cols),), dtype=np.float64)
    n_local     = np.array([Xtr.shape[0]], dtype=np.int64)

    sum_glob   = np.empty_like(sum_local)
    sumsq_glob = np.empty_like(sumsq_local)
    n_glob     = np.array([0], dtype=np.int64)

    comm.Allreduce(sum_local,   sum_glob,   op=MPI.SUM)
    comm.Allreduce(sumsq_local, sumsq_glob, op=MPI.SUM)
    comm.Allreduce(n_local,     n_glob,     op=MPI.SUM)

    n = int(n_glob[0]) if n_glob[0] > 0 else 1
    mu  = (sum_glob / n).astype("float32")
    var = (sumsq_glob / n) - (mu.astype(np.float64)**2)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var).astype("float32")
    std[std == 0.0] = 1.0

    Xtr = _sanitize((Xtr - mu) / std, cap=1e6)
    Xte = _sanitize((Xte - mu) / std, cap=1e6)

    if rank == 0:
        print(f"[Loader] feat_dim={len(feat_cols)}  train_local_rows={Xtr.shape[0]}  test_rows={Xte.shape[0]}")
    return Xtr, ytr, Xte, yte, feat_cols

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

def _act(z, kind):
    if kind == "relu":    return np.maximum(0.0, z)
    if kind == "tanh":    return np.tanh(z)
    if kind == "sigmoid": return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"Unknown activation '{kind}'")

def _act_grad(a, z, kind):
    if kind == "relu":    return (z > 0).astype(z.dtype)
    if kind == "tanh":    return 1.0 - a**2
    if kind == "sigmoid": return a * (1.0 - a)
    raise ValueError(f"Unknown activation '{kind}'")

class OneHiddenNN:
    def __init__(self, n_in, n_hidden, activation="tanh", seed=42):
        rng = np.random.default_rng(seed)
        lim1 = math.sqrt(6.0 / (n_in + n_hidden))
        self.W1 = rng.uniform(-lim1, lim1, size=(n_in, n_hidden)).astype("float32")
        self.b1 = np.zeros((n_hidden,), dtype="float32")
        lim2 = math.sqrt(6.0 / (n_hidden + 1))
        self.W2 = rng.uniform(-lim2, lim2, size=(n_hidden, 1)).astype("float32")
        self.b2 = np.zeros((1,), dtype="float32")
        self.activation = activation
        #self.gW1 = np.zeros_like(self.W1)
        #self.gb1 = np.zeros_like(self.b1)
        #self.gW2 = np.zeros_like(self.W2)
        #self.gb2 = np.zeros_like(self.b2)
        self.gW1 = np.zeros_like(self.W1, dtype="float32")
        self.gb1 = np.zeros_like(self.b1, dtype="float32")
        self.gW2 = np.zeros_like(self.W2, dtype="float32")
        self.gb2 = np.zeros_like(self.b2, dtype="float32")
    def forward(self, X):
        Z1 = X @ self.W1 + self.b1[None, :]
        A1 = _act(Z1, self.activation)
        Yhat = A1 @ self.W2 + self.b2[None, :]
        return Z1, A1, Yhat

    def loss_mse(self, Yhat, y):
        y = y.reshape(-1, 1)
        d = (Yhat - y)
        return 0.5 * float(np.mean(d * d)) if y.size else 0.0

    def backward(self, X, Z1, A1, Yhat, y):
        B = max(X.shape[0], 1)
        y = y.reshape(-1, 1)
        diff = (Yhat - y)
        self.gW2[:] = (A1.T @ diff) / B
        self.gb2[:] = np.mean(diff, axis=0)
        dA1 = diff @ self.W2.T
        dZ1 = dA1 * _act_grad(A1, Z1, self.activation)
        self.gW1[:] = (X.T @ dZ1) / B
        self.gb1[:] = np.mean(dZ1, axis=0)

    def apply(self, lr):
        self.W1 -= lr * self.gW1
        self.b1 -= lr * self.gb1
        self.W2 -= lr * self.gW2
        self.b2 -= lr * self.gb2

    def predict(self, X, batch=8192):
        if X.shape[0] == 0:
            return np.zeros((0,), dtype="float32")
        outs = []
        for i in range(0, X.shape[0], batch):
            _, _, yhat = self.forward(X[i:i+batch])
            outs.append(yhat)
        return np.vstack(outs).reshape(-1).astype("float32", copy=False)

# ------------------------------------------------------------
# MPI helpers & training
# ------------------------------------------------------------

def _mpi_avg_scalar(x: float, comm: MPI.Comm) -> float:
    buf = np.array([float(x)], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    buf /= comm.Get_size()
    return float(buf[0])

#def _allreduce_mean_inplace(arr: np.ndarray, comm: MPI.Comm):
    tmp = arr.astype(np.float64, copy=True)
    comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
    tmp /= comm.Get_size()
    arr[:] = tmp.astype(arr.dtype, copy=False)

def _allreduce_mean_inplace(arr, comm):
    """Safe allreduce that enforces matching shape/dtype and avoids IN_PLACE."""
    a = np.ascontiguousarray(arr.astype("float32", copy=False))   # ← NEW
    shapes = comm.allgather(a.shape)                               # ← NEW
    dtypes = comm.allgather(str(a.dtype))                          # ← NEW
    if len(set(shapes)) != 1 or len(set(dtypes)) != 1:             # ← NEW
        raise RuntimeError(f"Allreduce mismatch: shapes={shapes}, dtypes={dtypes}")  # ← NEW
    tmp = np.empty_like(a)                                         # ← NEW
    comm.Allreduce(a, tmp, op=MPI.SUM)                             # ← NEW
    arr[...] = (tmp / comm.size).astype(arr.dtype, copy=False)     # ← NEW

# ---------- Eval helper: local SSE & count for safe global RMSE ---------- #new
def _eval_local_stats(model, X_local: np.ndarray, y_local: np.ndarray, batch: int = 8192):
    """
    Compute local SSE and sample count for RMSE aggregation.
    Returns (sse_local: float64, n_local: float64)
    """
    X_local = np.asarray(X_local)
    y_local = np.asarray(y_local).reshape(-1)

    if X_local.size == 0 or y_local.size == 0:
        return np.float64(0.0), np.float64(0.0)

    sse = np.float64(0.0)
    n   = np.float64(0.0)

    for i in range(0, X_local.shape[0], int(batch)):
        xb = X_local[i:i+batch].astype("float32", copy=False)
        yb = y_local[i:i+batch].astype("float32", copy=False)

        pred = model.predict(xb, batch=xb.shape[0]).reshape(-1).astype("float32", copy=False)
        diff = (pred.astype("float64") - yb.astype("float64"))
        sse += np.dot(diff, diff)
        n   += np.float64(yb.size)

    return sse, n
# -----------------------------------------------------------------------

def _rmse_from_shards(model, X_local, y_local, comm: MPI.Comm, batch=8192):
    if X_local.shape[0] == 0:
        sse_local, n_local = 0.0, 0
    else:
        pred = model.predict(X_local, batch=batch)
        diff = (pred - y_local).astype("float64", copy=False)
        sse_local = float(np.dot(diff, diff))
        n_local = int(diff.shape[0])

    buf = np.array([sse_local, float(n_local)], dtype=np.float64)
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    tot_sse, tot_n = buf
    return math.sqrt(tot_sse / max(int(tot_n), 1))

def _train_sgd(model, X_local, y_local, *, lr=3e-4, batch=512, epochs=100,
               patience=20, comm=MPI.COMM_WORLD, log_every=5):
    X_local = X_local.astype("float32", copy=False)                            # ← NEW
    y_local = y_local.astype("float32", copy=False)                            # ← NEW
    rng = np.random.default_rng(123)
    rank = comm.Get_rank()

    hist, best_loss, best_snap, since = [], float("inf"), None, 0
    n_local = X_local.shape[0]

    for ep in range(1, epochs+1):
        log(f"Starting epoch {ep}")   #new
        t0 = time.time()
        if n_local > 0:
            order = rng.permutation(n_local)
            X_local = X_local[order]
            y_local = y_local[order]

        steps = (n_local + batch - 1) // batch if n_local > 0 else 1
        ep_loss_acc, ep_cnt = 0.0, 0

        for b in range(steps):
            if n_local == 0:
                model.gW1.fill(0.0); model.gb1.fill(0.0)
                model.gW2.fill(0.0); model.gb2.fill(0.0)
                local_loss = 0.0
            else:
                s = b * batch
                e = min(n_local, s + batch)
                Xb, yb = X_local[s:e], y_local[s:e]
                Z1, A1, Yhat = model.forward(Xb)
                local_loss = model.loss_mse(Yhat, yb)
                model.backward(Xb, Z1, A1, Yhat, yb)

            # Debug print once per rank at first epoch before sync:
            if ep == 1: #new
                log(f"gW1{model.gW1.shape} {model.gW1.dtype} | " #new
                    f"gb1{model.gb1.shape} {model.gb1.dtype} | " #new
                    f"gW2{model.gW2.shape} {model.gW2.dtype} | " #new
                    f"gb2{model.gb2.shape} {model.gb2.dtype}") #new
            
            log("Starting gradient sync") #new
            _allreduce_mean_inplace(model.gW1, comm)
            _allreduce_mean_inplace(model.gb1, comm)
            _allreduce_mean_inplace(model.gW2, comm)
            _allreduce_mean_inplace(model.gb2, comm)
            
            log("Finished gradient sync") #new
            for g in (model.gW1, model.gb1, model.gW2, model.gb2):
                np.clip(g, -1e2, 1e2, out=g)
                np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

            model.apply(lr)
            
            if ep == 1:                                             # ← NEW: one-time shapes print
                log(f"gW1{model.gW1.shape} {model.gW1.dtype} | "
                f"gb1{model.gb1.shape} {model.gb1.dtype} | "
                f"gW2{model.gW2.shape} {model.gW2.dtype} | "
                f"gb2{model.gb2.shape} {model.gb2.dtype}")
            log(f"Finished epoch {ep}")
            ep_loss_acc += _mpi_avg_scalar(local_loss, comm)
            ep_cnt += 1

        ep_loss = ep_loss_acc / max(ep_cnt, 1)
        hist.append(ep_loss)

        improved = ep_loss < best_loss - 1e-8
        since = 0 if improved else since + 1
        if improved:
            best_loss = ep_loss
            best_snap = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())

        if rank == 0 and (ep == 1 or ep % log_every == 0 or ep == epochs):
            print(f"[Epoch {ep:4d}] loss={ep_loss:.6f} (best {best_loss:.6f}) "
                  f"steps={ep_cnt} time={time.time()-t0:.2f}s")

        if patience is not None and since >= patience:
            if rank == 0:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs).")
            break

    if best_snap is not None:
        model.W1[:], model.b1[:], model.W2[:], model.b2[:] = best_snap

    return hist

# ------------------------------------------------------------
# Results I/O
# ------------------------------------------------------------

def _save_results(args, hist, tr_rmse, te_rmse, train_time, comm: MPI.Comm):
    if comm.Get_rank() != 0:
        return

    outdir = getattr(args, "outdir", "results/exp1")
    os.makedirs(outdir, exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "host": socket.gethostname(),
        "world": comm.Get_size(),
        "activation": args.act,
        "hidden": args.hidden,
        "batch": args.batch,
        "lr": (args.lr[0] if isinstance(args.lr, (list, tuple)) else args.lr),
        "epochs_run": len(hist),
        "train_rmse": float(tr_rmse),
        "test_rmse": float(te_rmse),
        "train_time_s": float(train_time),
    }
    outcsv = os.path.join(outdir, "results.csv")
    exists = os.path.isfile(outcsv)
    with open(outcsv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

    if getattr(args, "save_history", False):
        with open(os.path.join(outdir, f"{getattr(args, 'history_filename', 'loss_curve')}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","avg_batch_loss"])
            for i, v in enumerate(hist, 1):
                w.writerow([i, float(v)])

    if getattr(args, "plot_history", False):
        plt.figure(); plt.plot(range(1, len(hist)+1), hist)
        plt.xlabel("Epoch"); plt.ylabel("Avg batch loss"); plt.title("Training loss")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=120); plt.close()

    print(f"✓ Results saved in {outdir}")

# ------------------------------------------------------------
# Public entrypoint used by your run.py
# ------------------------------------------------------------

def main(args):
    """
    Expects a namespace with:
      train, test, ycol, hidden, act, lr, batch, epochs, patience,
      outdir, save_history, plot_history, (optional) history_filename
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    lr = args.lr[0] if isinstance(args.lr, (list, tuple)) else args.lr

    if rank == 0:
        print(f"[INFO] world={world} act={args.act} hidden={args.hidden} lr={lr} "
              f"batch={args.batch} epochs={args.epochs}")

    # --------- Load (sharded) ---------
    Xtr, ytr, Xte, yte, feat_cols = load_parquet_sharded(args.train, args.test, args.ycol, comm)
    d = len(feat_cols)
    # Enforce input dtypes once (prevents float64/float32 mismatches)           # ← NEW
    Xtr = Xtr.astype("float32", copy=False)                                    # ← NEW
    ytr = ytr.astype("float32", copy=False)                                    # ← NEW
    Xte = Xte.astype("float32", copy=False)                                    # ← NEW
    yte = yte.astype("float32", copy=False)                                    # ← NEW
    if rank == 0:
        print(f"[DATA] features={d}; local train rows={Xtr.shape[0]} | test rows (global)={yte.shape[0]}")

    # --------- Model ---------
    #model = OneHiddenNN(n_in=d, n_hidden=args.hidden, activation=args.act, seed=2025 + rank)
    model = OneHiddenNN(n_in=d, n_hidden=args.hidden, activation=args.act, seed=2025 + rank)
    # Normalize model params/grads to float32 across ranks (belt & suspenders)  # ← NEW
    for name in ("W1","b1","W2","b2","gW1","gb1","gW2","gb2"):                 # ← NEW
        a = getattr(model, name, None)                                         # ← NEW
        if a is not None:                                                      # ← NEW
            setattr(model, name, a.astype("float32", copy=False))              # ← NEW

    # --------- Train ---------
    t0 = time.time()
    hist = _train_sgd(model, Xtr, ytr, lr=lr, batch=args.batch, epochs=args.epochs,
                      patience=args.patience, comm=comm, log_every=5)
    #local_rmse = np.array(compute_rmse_local(...), dtype='float32') #new
    #global_rmse = np.empty_like(local_rmse) if rank == 0 else None #new
    #comm.Reduce(local_rmse, global_rmse, op=MPI.SUM, root=0) #new
    #if rank == 0: #new
    #    global_rmse /= comm.size
    #    log(f"RMSE={global_rmse}")
    #    save_history(hist, outdir)

    # ---- Post-training evaluation: global RMSE via Reduce([SSE, N]) ----
    sse_local, n_local = _eval_local_stats(model, Xte, yte, batch=8192)    # returns float64
    stats_local  = np.array([sse_local, n_local], dtype="float64")
    stats_global = np.empty_like(stats_local) if rank == 0 else np.empty(2, dtype="float64")

    comm.Reduce(stats_local, stats_global, op=MPI.SUM, root=0)

    if rank == 0:
        sse_tot, n_tot = float(stats_global[0]), float(stats_global[1])
        rmse = (sse_tot / max(n_tot, 1.0)) ** 0.5
        log(f"POST: global RMSE={rmse:.6f} (sse={sse_tot:.2f}, n={int(n_tot)})")
        # e.g., save history/results here on root
        # save_history(hist, args.outdir)
    # --------------------------------------------------------------------
    train_time = time.time() - t0

    # --------- Evaluate (distributed RMSE using shards) ---------
    tr_rmse = _rmse_from_shards(model, Xtr, ytr, comm)
    te_rmse = _rmse_from_shards(model, Xte, yte, comm)

    if rank == 0:
        print("\n=== Results ===")
        print(f"Activation     : {args.act}")
        print(f"Hidden units   : {args.hidden}")
        print(f"Batch size     : {args.batch}")
        print(f"Learning rate  : {lr}")
        print(f"Epochs run     : {len(hist)}")
        print(f"Train RMSE     : {tr_rmse:.6f}")
        print(f"Test  RMSE     : {te_rmse:.6f}")
        print(f"Train time (s) : {train_time:.2f}")

    _save_results(args, hist, tr_rmse, te_rmse, train_time, comm)