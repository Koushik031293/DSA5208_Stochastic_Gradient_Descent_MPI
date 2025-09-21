import argparse
import os
import sys
from time import time

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
import types

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DSS5208 Project Runner (MPI)")

    # Data
    ap.add_argument("--train", default="data/taxi_train.parquet", help="Path to TRAIN parquet")
    ap.add_argument("--test",  default="data/taxi_test.parquet",  help="Path to TEST parquet")
    ap.add_argument("--ycol",  default="total_amount",            help="Target column name")

    # Model hyperparameters
    ap.add_argument("--hidden",  type=int, default=64,                     help="Hidden units")
    ap.add_argument("--act",     choices=["relu","tanh","sigmoid"], default="relu", help="Activation")
    ap.add_argument("--lr",      type=float, default=1e-3,                 help="Learning rate")
    ap.add_argument("--batch",   type=int,   default=256,                  help="Mini-batch size")
    ap.add_argument("--epochs",  type=int,   default=120,                  help="Max epochs")
    ap.add_argument("--patience",type=int,   default=20,                   help="Early-stopping patience")

    # Outputs
    ap.add_argument("--outdir",        default="results/exp1", help="Directory to write artifacts")
    ap.add_argument("--save-history",  action="store_true",     help="Save loss_curve.csv")
    ap.add_argument("--plot-history",  action="store_true",     help="Save loss_curve.png & residuals_hist.png")

    # Sweep controls
    ap.add_argument("--sweep",    action="store_true", help="Run activation x batch grid on current MPI world")
    ap.add_argument("--acts",     default="relu,tanh,sigmoid", help="Comma-separated activations for sweep")
    ap.add_argument("--batches",  default="32,64,128,256,512", help="Comma-separated batch sizes for sweep")

    # Optional: merge all results.csv after sweep
    ap.add_argument("--merge-sweep", action="store_true", help="After sweep, merge sub-results into sweep_merged.csv")

    return ap.parse_args()


def make_ns(d):
    """Convenience: convert dict to SimpleNamespace for train_main."""
    return types.SimpleNamespace(**d)


# ------------------------------
# Runner
# ------------------------------
if __name__ == "__main__":
    args = parse_args()

    # Resolve paths relative to repo root
    args.train  = os.path.abspath(os.path.join(ROOT, args.train))
    args.test   = os.path.abspath(os.path.join(ROOT, args.test))
    args.outdir = os.path.abspath(os.path.join(ROOT, args.outdir))
    os.makedirs(args.outdir, exist_ok=True)

    if not args.sweep:
        # Single run
        t0 = time()
        train_main(args)
        dt = time() - t0
        print(f"\n✓ Finished single run in {dt:.2f}s. Artifacts: {args.outdir}")
    else:
        # Grid sweep: (activation x batch)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world = comm.Get_size()

        acts    = [s.strip() for s in args.acts.split(',') if s.strip()]
        batches = [int(s.strip()) for s in args.batches.split(',') if s.strip()]

        if rank == 0:
            print(f"Running sweep on world={world}:")
            print(f"  activations: {acts}")
            print(f"  batches    : {batches}")

        for a in acts:
            for b in batches:
                sub_out = os.path.join(args.outdir, f"act_{a}_b{b}")
                os.makedirs(sub_out, exist_ok=True)
                sub_args = make_ns({
                    "train": args.train,
                    "test": args.test,
                    "ycol": args.ycol,
                    "hidden": args.hidden,
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
                    print(f"\n=== Sweep: act={a}, batch={b} → {os.path.relpath(sub_out, ROOT)} ===")
                t0 = time()
                try:
                    train_main(sub_args)
                except Exception as ex:
                    # Keep sweep resilient; report and continue
                    if rank == 0:
                        print(f"[WARN] Sweep sub-run failed (act={a}, batch={b}): {ex}")
                finally:
                    if rank == 0:
                        print(f"  ↳ elapsed: {time()-t0:.2f}s")

        # Optional merge of results after sweep (rank 0 only)
        if args.merge_sweep:
            if rank == 0:
                merged_path = os.path.join(args.outdir, "sweep_merged.csv")
                import csv
                rows, header = [], None
                for dp, _, files in os.walk(args.outdir):
                    if "results.csv" in files and os.path.abspath(dp) != os.path.abspath(args.outdir):
                        path = os.path.join(dp, "results.csv")
                        with open(path, newline="") as f:
                            r = list(csv.DictReader(f))
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
