import argparse
import os
import sys
from src.DSA5208GP1 import main as train_main

# Ensure we can import from src/ or root (depending on user's layout)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT, "src")

if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
else:
    # If the script sits alongside DSA5208GP1.py (e.g., uploaded root), also try ROOT
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

try:
    from DSA5208GP1 import main as train_main
except Exception as e:
    print("Failed to import DSA5208GP1.main. Make sure DSA5208GP1.py is in ./src or project root.")
    raise

def parse_args():
    parser = argparse.ArgumentParser(description="Run Taxi Fare Prediction with MPI")
    parser.add_argument("--train", type=str, default="data/taxi_train.parquet",
                        help="Path to train parquet file (default: data/taxi_train.parquet)")
    parser.add_argument("--test", type=str, default="data/taxi_test.parquet",
                        help="Path to test parquet file (default: data/taxi_test.parquet)")
    parser.add_argument("--ycol", type=str, default="total_amount",
                        help="Target column name")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden layer size")
    parser.add_argument("--act", type=str, choices=["relu", "tanh", "sigmoid"], default="relu",
                        help="Activation function")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch", type=int, default=512,
                        help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max number of epochs")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience for early stopping")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Expand relative paths based on project root
    args.train = os.path.abspath(os.path.join(ROOT, args.train))
    args.test = os.path.abspath(os.path.join(ROOT, args.test))
    train_main(args)
