#!/usr/bin/env python3
import os, glob, re, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
def _ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _write_gallery(outdir, image_files, md_file="plots_per_activation.md", title="# Per-Activation Results"):
    md = os.path.join(outdir, md_file)
    with open(md, "w") as f:
        f.write(title + "\n\n")
        for img in image_files:
            rel_path = os.path.join("per_activation", img)  # relative path in md
            f.write(f"## {img}\n\n![{img}]({rel_path})\n\n")
    print("✓ Markdown gallery written:", md)

# ---------- (1) RMSE: lines & scatter per activation ----------
def plot_rmse_lines_per_activation(df, act, out_png):
    sub = df[df["activation"] == act].copy()
    if sub.empty: return None
    g = (sub.groupby("world", as_index=False)
             .agg(train_rmse=("train_rmse","mean"),
                  test_rmse=("test_rmse","mean"))
             .sort_values("world"))
    plt.figure()
    plt.plot(g["world"], g["train_rmse"], marker="o", linestyle="-", label="train")
    plt.plot(g["world"], g["test_rmse"],  marker="x", linestyle="--", label="test")
    plt.xlabel("MPI processes (world size)")
    plt.ylabel("RMSE")
    plt.title(f"Train vs Test RMSE vs processes — {act}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return os.path.basename(out_png)

def plot_rmse_scatter_per_activation(df, act, out_png):
    sub = df[df["activation"] == act].copy()
    if sub.empty: return None
    plt.figure()
    for w, dsub in sub.groupby("world"):
        plt.scatter(dsub["train_rmse"], dsub["test_rmse"],
                    s=50, alpha=0.8, label=f"world={int(w)}")
    vmin = float(min(sub["train_rmse"].min(), sub["test_rmse"].min()))
    vmax = float(max(sub["train_rmse"].max(), sub["test_rmse"].max()))
    plt.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
    plt.xlabel("Train RMSE"); plt.ylabel("Test RMSE")
    plt.title(f"Train vs Test RMSE (parity) — {act}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return os.path.basename(out_png)

# ---------- (2) Train time vs processes (all activations on one chart) ----------
def plot_time_all_activations(df, out_png):
    if "activation" not in df.columns:
        print("No activation column in results.csv; skipping training-time plot.")
        return None
    plt.figure()
    for act, sub in df.groupby("activation"):
        g = (sub.groupby("world", as_index=False)["train_time_s"]
               .mean().sort_values("world"))
        plt.plot(g["world"], g["train_time_s"], marker="o", label=act)
    plt.xlabel("MPI processes (world size)")
    plt.ylabel("Train time (s)")
    plt.title("Training time vs processes (per activation)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    print(f"Saved combined training time vs processes -> {out_png}")
    return os.path.basename(out_png)

# ---------- (3) Loss curves across processes for a fixed activation ----------
def plot_loss_curves_by_process(search_dir, act, out_png):
    """
    Expects history files named like:
      loss_<ACT>_world{p}_batch{b}.csv   OR
      loss_world{p}_batch{b}_{ACT}.csv
    with columns epoch,avg_batch_loss.
    """
    files = glob.glob(os.path.join(search_dir, "loss_*world*batch*.csv"))
    keep = [f for f in files if act.lower() in os.path.basename(f).lower()]
    curves = {}
    for f in keep:
        m = re.search(r"world(\d+)", os.path.basename(f))
        if not m: 
            continue
        w = int(m.group(1))
        try:
            hist = pd.read_csv(f)
            if {"epoch","avg_batch_loss"}.issubset(hist.columns):
                curves[w] = (hist["epoch"], hist["avg_batch_loss"])
        except Exception as e:
            print(f"Skip {f}: {e}")
    if not curves: 
        return None
    plt.figure()
    for w, (x,y) in sorted(curves.items()):
        plt.plot(x, y, marker="o", label=f"world={w}")
    plt.xlabel("Epoch"); plt.ylabel("R(θk)  (avg batch loss)")
    plt.title(f"Training history (R(θk) vs k) — {act}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()
    return os.path.basename(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.csv")
    ap.add_argument("--outdir",  required=True, help="Where plots & markdown go")
    args = ap.parse_args()

    # Create subdir for images
    imgdir = os.path.join(args.outdir, "per_activation")
    os.makedirs(imgdir, exist_ok=True)

    df = pd.read_csv(args.results)
    _ensure_numeric(df, ["world","batch","train_rmse","test_rmse","train_time_s"])
    df = df.dropna(subset=["world","batch","train_rmse","test_rmse","train_time_s","activation"])

    images = []
    activations = sorted(df["activation"].dropna().unique())
    for act in activations:
        # (1) RMSE line & scatter per activation
        img1 = plot_rmse_lines_per_activation(df, act, os.path.join(imgdir, f"rmse_lines_{act}.png"))
        img2 = plot_rmse_scatter_per_activation(df, act, os.path.join(imgdir, f"rmse_scatter_{act}.png"))
        # (3) Loss curves across processes for this activation
        img3 = plot_loss_curves_by_process(args.outdir, act, os.path.join(imgdir, f"loss_curves_{act}.png"))
        for im in (img1, img2, img3):
            if im: images.append(im)

    # (2) ONE combined training-time chart for all activations
    time_img = plot_time_all_activations(df, os.path.join(imgdir, "time_vs_procs_by_activation.png"))
    if time_img:
        images.append(time_img)

    _write_gallery(args.outdir, images, md_file="plots_per_activation.md")
    print("✓ Done.", len(images), "images stored in", imgdir)

if __name__ == "__main__":
    main()