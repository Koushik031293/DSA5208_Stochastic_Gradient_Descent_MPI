# DSA5208_Stochastic_Gradient_Descent_MPI



## ⚙️ Setup

You can set up the environment in **two ways**:  

Option 1: pip

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt

Option 2: conda (recommended, ensures MPI works)
conda env create -f environment.yml
conda activate mpi-sgd-env

**Steps to run each stages**

**Preprocessing: **

python src/data_preprocessing.py \
  --csv data/nytaxi2022.csv \
  --outdir data \
  --ycol total_amount \
  --seed 42 \
  --emit-combined \
  --save-csv \
  --scaler-npz data/scaler_numeric_v1.npz \
  --meta-json data/feature_meta_v1.json

**Train Test MPI**

mpiexec -n 4 python main.py \
  --train data/taxi_train.parquet \
  --test  data/taxi_test.parquet \
  --ycol total_amount \
  --hidden 32 \
  --act relu \
  --lr 1e-3 \
  --batch 128 \
  --epochs 20 \
  --outdir results/debug \
  --save-history \
  --plot-history


**Sweep Function : **

mpiexec -n 4 python main.py --sweep \
  --train data/taxi_train.parquet \
  --test  data/taxi_test.parquet \
  --ycol total_amount \
  --hidden 32 --lr 1e-3 --epochs 40 --patience 10 \
  --acts relu,tanh \
  --batches 64,128,256 \
  --outdir results/sweep_run \
  --save-history --plot-history --merge-sweep



mpiexec -n 4 python main.py --sweep \
  --train data/taxi_train.parquet \
  --test  data/taxi_test.parquet \
  --ycol total_amount \
  --lr 1e-3 --epochs 40 --patience 10 \
  --acts relu,tanh,sigmoid \
  --batches 32,64,128,256,512 \
  --random-hidden --hidden-min 32 --hidden-max 256 --seed 123 \
  --outdir results/sweep_run \
  --save-history --plot-history --merge-sweep
