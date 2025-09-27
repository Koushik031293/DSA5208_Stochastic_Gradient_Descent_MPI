#!/usr/bin/env python3
import argparse
import os
import sys
from time import time
import types

# ------------------------------
# Import project training entry
# ------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_mpi_sgd import main as train_main


# ------------------------------
# CLI
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="DSS5208 Project Runner (MPI)")

    # Data
    ap.add_argument("--train", default="data/taxi_train.parquet", help="TRAIN parquet")
    ap.add_argument("--test",  default="data/taxi_test.parquet",  help="TEST parquet")
    ap.add_argument("--ycol",  default="total_amount",            help="prediction column")

    # Model hyperparameters
    ap.add_argument("--hidden",   type=int, default=64, help="Hidden units")
    ap.add_argument("--act",      choices=["relu","tanh","sigmoid"], default="relu", help="Activation function")
    def comma_list_floats(x):
        return [float(v) for v in x.split(",")]

    ap.add_argument("--lr", type=comma_list_floats, default=[1e-3],
                help="Comma-separated learning rates")
    ap.add_argument("--batch",    type=int,   default=256, help="Mini-batch size")
    ap.add_argument("--epochs",   type=int,   default=120, help="Max epochs")
    ap.add_argument("--patience", type=int,   default=20,  help="Early-stopping patience")

    # Outputs
    ap.add_argument("--outdir",        default="results/exp1", help="output directory")
    ap.add_argument("--save-history",  action="store_true",    help="Save loss_curve.csv")
    ap.add_argument("--plot-history",  action="store_true",    help="Save loss_curve.png & residuals_hist.png")

    # Sweep
    ap.add_argument("--sweep",    action="store_true", help="Run activation × batch grid on current MPI world")
    ap.add_argument("--acts",     default="relu,tanh,sigmoid", help="Comma-separated activations for sweep")
    ap.add_argument("--batches",  default="32,64,128,256,512", help="Comma-separated batch sizes for sweep")

    # Random hidden layer (for sweep only)
    ap.add_argument("--random-hidden", action="store_true",
                    help="During --sweep, sample hidden units randomly per (activation,batch) sub-run")
    ap.add_argument("--hidden-min", type=int, default=32,
                    help="Min hidden units when --random-hidden is on")
    ap.add_argument("--hidden-max", type=int, default=256,
                    help="Max hidden units when --random-hidden is on (inclusive)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducible random hidden sizes in sweep")

    # Merge all sub-results.csv after sweep
    ap.add_argument("--merge-sweep", action="store_true")
    ap.add_argument("--history-filename", type=str, default="loss_curve")

    return ap.parse_args()


def make_ns(d):
    """Convert dict to SimpleNamespace for train_main."""
    return types.SimpleNamespace(**d)


if __name__ == "__main__":
    args = parse_args()

    # Resolve paths relative to repo root
    args.train  = os.path.abspath(os.path.join(ROOT, args.train))
    args.test   = os.path.abspath(os.path.join(ROOT, args.test))
    args.outdir = os.path.abspath(os.path.join(ROOT, args.outdir))
    os.makedirs(args.outdir, exist_ok=True)

    if not args.sweep:
        # --------------------------
        # Single run
        # --------------------------
        t0 = time()
        train_main(args)
        dt = time() - t0
        print(f"\n✓ Finished single run in {dt:.2f}s. Artifacts: {args.outdir}")
    else:
        
        # Grid sweep: (activation × batch)
        
        from mpi4py import MPI
        import random

        comm  = MPI.COMM_WORLD
        rank  = comm.Get_rank()
        world = comm.Get_size()

        acts    = [s.strip() for s in args.acts.split(",") if s.strip()]
        batches = [int(s.strip()) for s in args.batches.split(",") if s.strip()]

        if rank == 0:
            print(f"Running sweep on world={world}:")
            print(f"  activations: {acts}")
            print(f"  batches    : {batches}")
            # Seed once on rank 0 so the sequence of hidden samples is reproducible
            random.seed(args.seed)

        for a in acts:
            for b in batches:
                # Decide hidden size (rank 0 samples, then broadcast)
                if args.random_hidden:
                    h_local = random.randint(args.hidden_min, args.hidden_max) if rank == 0 else None
                    h = comm.bcast(h_local, root=0)
                else:
                    h = args.hidden

                # Subdir name includes hidden size for traceability
                sub_out = os.path.join(args.outdir, f"act_{a}_b{b}_h{h}")
                os.makedirs(sub_out, exist_ok=True)

                sub_args = make_ns({
                    "train": args.train,
                    "test": args.test,
                    "ycol": args.ycol,
                    "hidden": h,
                    "act": a,
                    "lr": args.lr,
                    "batch": b,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "outdir": sub_out,
                    "save_history": True if args.save_history or args.plot_history else False,
                    "plot_history": True if args.plot_history else False,
                })

                if rank == 0:
                    rel = os.path.relpath(sub_out, ROOT)
                    print(f"\n=== Sweep: act={a}, batch={b}, hidden={h} → {rel} ===")

                t0 = time()
                try:
                    train_main(sub_args)
                except Exception as ex:
                    # Keep sweep resilient; report and continue
                    if rank == 0:
                        print(f"[WARN] Sweep sub-run failed (act={a}, batch={b}, hidden={h}): {ex}")
                finally:
                    if rank == 0:
                        print(f"  ↳ elapsed: {time()-t0:.2f}s")

    
        # Optional merge (rank 0)
        if args.merge_sweep:
            if rank == 0:
                merged_path = os.path.join(args.outdir, "sweep_merged.csv")
                import csv
                rows, header = [], None
                for dp, _, files in os.walk(args.outdir):
                    # Skip the root outdir; only read leaf subfolders
                    if "results.csv" in files and os.path.abspath(dp) != os.path.abspath(args.outdir):
                        path = os.path.join(dp, "results.csv")
                        try:
                            with open(path, newline="") as f:
                                r = list(csv.DictReader(f))
                        except Exception as e:
                            print(f"[WARN] Could not read {path}: {e}")
                            continue
                        if not r:
                            continue
                        for row in r:
                            row["subdir"] = os.path.relpath(dp, args.outdir)
                            rows.append(row)
                        if header is None:
                            header = list(r[0].keys()) + ["subdir"]
                if rows and header:
                    with open(merged_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=header)
                        w.writeheader(); w.writerows(rows)
                    print(f"\n✓ Sweep complete. Merged: {merged_path}")
                else:
                    print("\nSweep complete. No sub-results found to merge.")
        else:
            if rank == 0:
                print("\n✓ Sweep complete. Consolidate results from each subfolder's results.csv if needed.")