#!/usr/bin/env python3
import argparse
import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser(description="Generate DSA5208 report figures: loss curves, RMSE, and training time scaling.")
    ap.add_argument("--summary", required=True, help="Path to summary CSV of runs.")
    ap.add_argument("--loss_dir", required=True, help="Directory containing per-run loss CSVs named like loss_<activation>_bs<batch>_w<world>.csv")
    ap.add_argument("--outdir", default="report_figs", help="Output directory for figures and aggregates")
    ap.add_argument("--activations", default="relu,tanh,sigmoid", help="Comma list to filter activations (default: relu,tanh,sigmoid)")
    ap.add_argument("--worlds", default="1,4", help="Comma list to filter MPI process counts (world sizes). Example: 1,4")
    ap.add_argument("--batches", default="", help="Optional comma list to filter batch sizes. Example: 32,64,128,256,512")
    return ap.parse_args()

def safe_float(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if x is None:
        return np.nan
    s = str(x).strip().strip("[](){}")
    try:
        return float(s)
    except Exception:
        return np.nan

def load_summary(path, acts, worlds, batches):
    df = pd.read_csv(path)
    # Normalize
    if "activation" in df.columns:
        df["activation"] = df["activation"].astype(str).str.lower().str.strip()
    if "world" in df.columns:
        df["world"] = pd.to_numeric(df["world"], errors="coerce").astype("Int64")
    if "batch" in df.columns:
        df["batch"] = pd.to_numeric(df["batch"], errors="coerce").astype("Int64")
    # normalize epoch column name if present
    if "epochs_run" in df.columns and "epochs" not in df.columns:
        df = df.rename(columns={"epochs_run": "epochs"})
    # lr cleanup
    if "lr" in df.columns:
        df["lr_float"] = df["lr"].apply(safe_float)
    # Filters
    if "activation" in df.columns:
        df = df[df["activation"].isin(acts)]
    if len(worlds) > 0 and "world" in df.columns:
        df = df[df["world"].isin(worlds)]
    if len(batches) > 0 and "batch" in df.columns:
        df = df[df["batch"].isin(batches)]
    return df

LOSS_RE = re.compile(r"loss_(?P<act>[a-zA-Z0-9]+)_bs(?P<batch>\d+)_w(?P<world>\d+)\.csv$")

def scan_loss_files(loss_dir, acts, worlds, batches):
    """Return mapping: (act, world) -> list of (batch, path) for allowed files."""
    mapping = defaultdict(list)
    for fname in os.listdir(loss_dir):
        m = LOSS_RE.match(fname)
        if not m:
            continue
        act = m.group("act").lower()
        batch = int(m.group("batch"))
        world = int(m.group("world"))
        if act in acts and world in worlds and (not batches or batch in batches):
            mapping[(act, world)].append((batch, os.path.join(loss_dir, fname)))
    # sort by batch for reproducible legend order
    for k in mapping:
        mapping[k].sort(key=lambda t: t[0])
    return mapping

def ensure_out(outdir):
    os.makedirs(outdir, exist_ok=True)

# ---------------- Existing R(theta_k) loss curves (avg_batch_loss) ----------------
def compute_global_loss_ylim(loss_groups):
    """
    Scan all loss CSVs referenced by loss_groups and return a common (ymin, ymax)
    for the 'avg_batch_loss' column across all files. Adds a small padding.
    """
    ymin, ymax = np.inf, -np.inf
    for (act, world), items in loss_groups.items():
        for batch, path in items:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
                continue
            if "avg_batch_loss" not in df.columns:
                print(f"[WARN] {path} missing 'avg_batch_loss'; skipping for y-limits.")
                continue
            y = pd.to_numeric(df["avg_batch_loss"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if y.empty:
                continue
            ymin = min(ymin, float(y.min()))
            ymax = max(ymax, float(y.max()))

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        return None  # fallback: let matplotlib autoscale if nothing valid found

    # add ~5% padding
    rng = ymax - ymin
    pad = 0.05 * (rng if rng > 0 else max(1e-6, abs(ymin)))
    return (ymin - pad, ymax + pad)

def plot_loss_curves(loss_groups, outdir, ylims=None):
    """One chart per (activation, world) with multiple batch curves. y=avg_batch_loss"""
    for (act, world), items in loss_groups.items():
        if not items:
            continue
        plt.figure(figsize=(8,5))
        for batch, path in items:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
                continue
            if not {"epoch","avg_batch_loss"}.issubset(df.columns):
                print(f"[WARN] {path} missing required columns epoch, avg_batch_loss")
                continue
            y = pd.to_numeric(df["avg_batch_loss"], errors="coerce").values
            x = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64").astype(int).values
            # Smooth for readability
            if len(y) >= 3:
                y_smooth = pd.Series(y).rolling(window=3, min_periods=1).mean().values
            else:
                y_smooth = y
            plt.plot(x, y_smooth, marker="o", label=f"bs{batch}")

        if ylims is not None and all(np.isfinite(ylims)):
            plt.ylim(ylims)

        plt.xlabel("Epoch (k)")
        plt.ylabel("R(θk) / Loss")
        plt.title(f"Training History: {act} (world={world})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Batch")
        plt.tight_layout()
        out_path = os.path.join(outdir, f"loss_{act}_w{world}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Saved {out_path}")


# -------- NEW: Time (Y) vs Epoch (X), one figure per (activation, world) ----------
def plot_time_vs_epoch(loss_groups, df_sum, outdir):
    """
    For each (activation, world), plot cumulative time vs epoch,
    with one line per batch. If per-epoch time is present in loss CSV
    (columns 'time_s' or 'epoch_time_s'), use it. Otherwise estimate:
      per_epoch = train_time_s / epochs  (from summary)
    """
    # Index summary by (act, world, batch)
    key_cols = ["activation", "world", "batch"]
    have_cols = set(df_sum.columns)
    for c in key_cols + ["train_time_s", "epochs"]:
        if c not in have_cols:
            print(f"[WARN] Summary missing '{c}'. Time-vs-epoch plots may be skipped.")
    idx = {}
    for _, r in df_sum.iterrows():
        key = (str(r.get("activation")).lower(), int(r.get("world")), int(r.get("batch")))
        idx[key] = {"train_time_s": r.get("train_time_s"), "epochs": r.get("epochs")}

    for (act, world), items in loss_groups.items():
        if not items:
            continue
        plt.figure(figsize=(8,5))
        plotted_any = False
        for batch, path in items:
            # Try to read loss with per-epoch timing
            df = None
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"[WARN] Could not read {path}: {e}")
                df = None

            x = None
            y_cum = None

            if df is not None and "epoch" in df.columns:
                x = df["epoch"].astype(int).values

                # Prefer explicit per-epoch timing if present
                per_epoch = None
                if "time_s" in df.columns:
                    per_epoch = df["time_s"].astype(float).values
                elif "epoch_time_s" in df.columns:
                    per_epoch = df["epoch_time_s"].astype(float).values

                if per_epoch is not None:
                    y_cum = np.cumsum(per_epoch)
                else:
                    # Fallback: estimate from summary (total / epochs) and build cumulative
                    key = (act, int(world), int(batch))
                    meta = idx.get(key)
                    if meta and pd.notna(meta.get("train_time_s")) and pd.notna(meta.get("epochs")) and int(meta["epochs"]) > 0:
                        per = float(meta["train_time_s"]) / int(meta["epochs"])
                        y_cum = per * x  # cumulative time assuming roughly uniform epoch time
                    else:
                        print(f"[WARN] No timing info for {key}; skipping in time-vs-epoch.")
                        continue
            else:
                print(f"[WARN] {path} missing 'epoch' column; skipping in time-vs-epoch.")
                continue

            plt.plot(x, y_cum, marker="o", label=f"bs{batch}")
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Epoch (k)")
        plt.ylabel("Cumulative Time (s)")
        plt.title(f"Time vs Epoch: {act} (world={world})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Batch")
        plt.tight_layout()
        out_path = os.path.join(outdir, f"time_vs_epoch_{act}_w{world}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Saved {out_path}")

# ---------------------- Existing grouped RMSE figure ----------------------
def compute_global_rmse_limits(df):
    """
    Returns a common (lo, hi) for both axes across all RMSE points in df.
    Uses finite values from train_rmse and test_rmse, with small padding.
    """
    req = {"train_rmse","test_rmse"}
    if not req.issubset(df.columns):
        return None
    x = pd.to_numeric(df["train_rmse"], errors="coerce")
    y = pd.to_numeric(df["test_rmse"], errors="coerce")
    vals = pd.concat([x, y], axis=0).replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    rng = hi - lo
    pad = 0.05 * (rng if rng > 0 else max(1e-6, abs(lo)))
    return (lo - pad, hi + pad)

def plot_rmse(df, outdir):
    if not {"train_rmse","test_rmse","activation","batch","world"}.issubset(df.columns):
        print("[WARN] Summary CSV missing required RMSE columns; skipping RMSE figure.")
        return
    df_plot = df.copy()
    df_plot["config"] = (df_plot["activation"].astype(str)
                         + "|bs" + df_plot["batch"].astype(str)
                         + "|w" + df_plot["world"].astype(str))
    df_plot = df_plot.sort_values(["activation","world","batch"])
    x = np.arange(len(df_plot))
    width = 0.4
    plt.figure(figsize=(max(10, len(df_plot)*0.4), 6))
    plt.bar(x - width/2, df_plot["train_rmse"].astype(float).values, width, label="Train RMSE")
    plt.bar(x + width/2, df_plot["test_rmse"].astype(float).values, width, label="Test RMSE")
    plt.xticks(x, df_plot["config"].tolist(), rotation=60, ha="right")
    plt.ylabel("RMSE")
    plt.title("Train vs Test RMSE by Configuration")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outdir, "rmse_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {out_path}")

# -------- NEW: 6 RMSE line graphs (per activation & world), x=batch ----------
# -------- MODIFIED: RMSE graphs (test=line, train=scatter) ----------
# -------- MODIFIED: RMSE graphs (x=train_rmse, y=test_rmse, line=Batch) ----------
# -------- UPDATED: Scatter of Test vs Train RMSE with y=x line ----------
# -------- UPDATED: Scatter of Test vs Train RMSE with y=x line ----------
def plot_rmse_per_act_world(df, outdir, xy_limits=None):
    """
    For each activation & world:
      - x-axis: Train RMSE
      - y-axis: Test RMSE
      - Points: one per (batch/config)
      - Dashed line: y = x reference
      - If xy_limits provided as (lo, hi), use it for both axes on all plots
    """
    req = {"activation","world","batch","train_rmse","test_rmse"}
    if not req.issubset(df.columns):
        print(f"[WARN] Summary CSV missing required columns {req}; skipping RMSE scatter plots.")
        return

    groups = df.groupby(["activation","world"], dropna=False)
    for (act, world), g in groups:
        if g.empty:
            continue

        x = pd.to_numeric(g["train_rmse"], errors="coerce").values
        y = pd.to_numeric(g["test_rmse"], errors="coerce").values

        plt.figure(figsize=(7,6))
        plt.scatter(x, y, s=50)

        # Determine common limits
        if xy_limits is not None and all(np.isfinite(xy_limits)):
            lo, hi = xy_limits
        else:
            # fallback to per-figure limits (if no global limits)
            lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
            hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
            pad = 0.02 * (hi - lo if hi > lo else max(1e-6, abs(lo)))
            lo, hi = lo - pad, hi + pad

        # y=x reference and axis settings
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.gca().set_aspect("equal", adjustable="box")

        plt.xlabel("Train RMSE")
        plt.ylabel("Test RMSE")
        plt.title(f"Train vs Test RMSE: {act} (world={int(world)})")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        out_path = os.path.join(outdir, f"rmse_scatter_{act}_w{int(world)}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Saved {out_path}")




# ------------------- Training time vs processes (existing) -------------------
def plot_training_time_vs_world(df, outdir):
    if not {"world","train_time_s"}.issubset(df.columns):
        print("[WARN] Summary CSV missing training time columns; skipping training time figure.")
        return
    grp = df.groupby("world")["train_time_s"].agg(["mean","std","count"]).reset_index().sort_values("world")
    # Plot mean with error bars (std)
    plt.figure(figsize=(7,5))
    plt.errorbar(grp["world"].astype(int).values, grp["mean"].astype(float).values,
                 yerr=grp["std"].fillna(0).astype(float).values, marker="o", linestyle="-")
    plt.xlabel("Processes (world)")
    plt.ylabel("Training Time (s)")
    plt.title("Training Time vs Number of Processes")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path = os.path.join(outdir, "training_time_vs_processes.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {out_path}")
    # Also save aggregates with speedup/efficiency if world=1 present
    t1 = float(grp.loc[grp["world"]==1, "mean"].values[0]) if (grp["world"]==1).any() else np.nan
    grp["speedup"] = t1 / grp["mean"] if not math.isnan(t1) else np.nan
    grp["efficiency"] = grp["speedup"] / grp["world"]
    agg_path = os.path.join(outdir, "aggregates.csv")
    grp.to_csv(agg_path, index=False)
    print(f"[OK] Saved {agg_path}")

# --------------------------------- MAIN ---------------------------------
def main():
    args = parse_args()
    ensure_out(args.outdir)
    acts = [s.strip().lower() for s in args.activations.split(",") if s.strip()]
    worlds = [int(s) for s in args.worlds.split(",") if s.strip()]
    batches = [int(s) for s in args.batches.split(",") if s.strip()] if args.batches else []

    # Load and filter summary
    df_sum = load_summary(args.summary, acts, worlds, batches)
    # RMSE figures
    if len(df_sum) > 0:
        # Existing grouped RMSE across all configs
        plot_rmse(df_sum, args.outdir)

        # Compute common limits for all RMSE scatter charts
        global_rmse_limits = compute_global_rmse_limits(df_sum)
        if global_rmse_limits is None:
            print("[INFO] Could not determine common RMSE limits. Using per-figure autoscale.")
        else:
            print(f"[INFO] Using common RMSE limits: {global_rmse_limits}")

        # Per-activation/per-world RMSE scatter with unified limits
        plot_rmse_per_act_world(df_sum, args.outdir, xy_limits=global_rmse_limits)

    # Loss files (R(theta_k) curves)
    loss_groups = scan_loss_files(args.loss_dir, acts, worlds, batches)
    if len(loss_groups) == 0:
        print("[WARN] No matching loss-curve files found. Ensure filenames follow: loss_<activation>_bs<batch>_w<world>.csv")
    else:
        # Compute a global y-limit for ALL loss-curve charts so they share the same scale
        common_loss_ylims = compute_global_loss_ylim(loss_groups)
        if common_loss_ylims is None:
            print("[INFO] Could not determine common y-limits (no valid loss values). Using autoscale.")
        else:
            print(f"[INFO] Using common loss y-limits: {common_loss_ylims}")

        # Existing R(θk) loss curves with unified y-axis
        plot_loss_curves(loss_groups, args.outdir, ylims=common_loss_ylims)

        # NEW: Time vs Epoch (unchanged; separate y-axis)
        if len(df_sum) > 0:
            plot_time_vs_epoch(loss_groups, df_sum, args.outdir)

    # Training time vs processes (existing)
    if len(df_sum) > 0:
        plot_training_time_vs_world(df_sum, args.outdir)

if __name__ == "__main__":
    main()

