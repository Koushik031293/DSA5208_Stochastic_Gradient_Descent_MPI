# 📘 DSA5208 – Stochastic Gradient Descent with MPI

## 📌 Project Overview
This project implements a **parallel Stochastic Gradient Descent (SGD)** training loop using **MPI** via the `mpi4py` Python library.  

- **Language:** Python 3  
- **Parallelism:** MPI (`mpi4py`)  
- **Dataset format:** Parquet (`.parquet`)  

---
## Dataset Access
The datasets are not included in this repository due to their size. Raw dataset can be found on kaggle (https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers/data). Cleansed and train/test dataset can be downloaded from OneDrive: https://nusu-my.sharepoint.com/:f:/r/personal/e1352339_u_nus_edu/Documents/DSA5208%20Scalable%20Distributed%20Computing%20for%20Data%20Science/Data?csf=1&web=1 

## Assumptions
- The `trip_distance` column is treated as **kilometres (km)**, not miles.  
- Filtering thresholds for speed, fare per km, and fare per minute were chosen accordingly.  
- Reference: [NYC TLC Data Dictionary – Yellow Taxi Trip Records](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)

## Data Cleaning Criteria
The following rules were applied before splitting:
- Dropped rows with missing values in required columns.
- Dropped rows with passengers_count, trip_distance, extra < **0**.   
- Dropped rows with duration <**1minute**.  
- Speed sanity: **1 ≤ km/h ≤ 120**.  
- Dropped rows with total amount < **\$3**.  
- RatecodeID restricted to `{1,2,3,4,5,6,99}`.  
- Fare-per-km and fare-per-min thresholds (different bounds for each `RatecodeID`):  
  - *1 Standard:* 0.5–20 $/km, 0.10–10 $/min  
  - *2 JFK:* 0.5–10 $/km, 0.03–10 $/min  
  - *3 Newark:* 0.5–12 $/km, 0.05–12 $/min  
  - *4 Nassau/Westchester:* 0.5–20 $/km, 0.05–12 $/min  
  - *5 Negotiated:* 0.05–30 $/km, 0.02–20 $/min  
  - *6 Group rides:* 0.05–20 $/km, 0.02–12 $/min  
  - *99 Unknown:* 0.05–30 $/km, 0.05–20 $/min  

## Train/Test Split
Data was split into **70% training /30% test** using **deterministic hash-based split**. This ensures reproducibility across machines and MPI workers. 

---

## 📌 Requirements
- Python ≥ 3.9  
- mpi4py ≥ 3.1  
- pandas, numpy, matplotlib  
---


## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourname/DSA5208_Stochastic_Gradient_Descent_MPI.git
   cd DSA5208_Stochastic_Gradient_Descent_MPI
   ```

2. Create environment & install dependencies:
   ```bash
   conda create -n dsa5208 python=3.11 -y
   conda activate dsa5208
   pip install -r requirements.txt
   ```

3. Ensure MPI is installed on your machine:
   ```bash
   # For macOS
   brew install open-mpi

   # For Ubuntu / Debian
   sudo apt-get install libopenmpi-dev openmpi-bin
   ```

4. Verify installation:
   ```bash
   mpirun --version
   ```

---

## 📂 Repository Structure
```
.
├── data/                 # Training & test parquet files
├── src/
│   ├── data_preprocessing.py           # Data preprocessing / split
│   ├── train_mpi_sgd.py                # SGD training with MPI
│   ├── main.py                         # Main entry point (handles args, runs training)
│   └── report_plots_per_activation.py  # Per-activation gallery
├── results/                    # Metrics CSVs, loss histories
├── output/                     # Final plots / reports
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Code

### 1. Preprocess the dataset
```bash
python src/data_preprocessing.py \
  --csv data/nytaxi2022.csv \
  --outdir data \
  --ycol total_amount \
  --seed 42 \
  --emit-combined \
  --save-csv \
  --scaler-npz data/scaler_numeric_v1.npz \
  --meta-json data/feature_meta_v1.json
```

### 2. Train with MPI
Run with **4 processes** on the taxi dataset:
```bash
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

```

### 3. Sweep Experiments
Grid search over activations and batch sizes:
```bash
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
```


### 4. Process Experiments
run the mpi function with different process , repeat the same code with change in process
```bash
mpiexec -n 1 python main.py \
  --train data/taxi_train.parquet \
  --test  data/taxi_test.parquet \
  --ycol total_amount \
  --act tanh --hidden 128 --lr 3e-4 --batch 256 \
  --epochs 40 --patience 10 \
  --outdir results/scaling --save-history

```

### 4. Best in Class Experiments
  ## Relu
```bash
BATCHES=(32 64 128 256 512)
PROCS=(1 5)    

for p in "${PROCS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    RUN_ID="p${p}_relu_h228_bs${batch}_$(date +%Y%m%d_%H%M%S)"
    echo "Running: procs=${p}, act=relu, hidden=228, batch=${batch}"
    mpiexec -n ${p} python main.py \
      --train data/taxi_train.parquet \
      --test  data/taxi_test.parquet \
      --ycol total_amount \
      --act relu --hidden 228 --lr 3e-4 --batch ${batch} \
      --epochs 40 --patience 10 \
      --outdir results/scaling/${RUN_ID} --save-history
    sleep 1
  done
done

```

  ## tanh
```bash
BATCHES=(32 64 128 256 512)
PROCS=(1 5)    

for p in "${PROCS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    RUN_ID="p${p}_tanh_h100_bs${batch}_$(date +%Y%m%d_%H%M%S)"
    echo "Running: procs=${p}, act=relu, hidden=228, batch=${batch}"
    mpiexec -n ${p} python main.py \
      --train data/taxi_train.parquet \
      --test  data/taxi_test.parquet \
      --ycol total_amount \
      --act tanh --hidden 100 --lr 3e-4 --batch ${batch} \
      --epochs 40 --patience 10 \
      --outdir results/scaling/${RUN_ID} --save-history
    sleep 1
  done
done

```
 ## sigmoid
```bash
BATCHES=(32 64 128 256 512)
PROCS=(1 5)    

for p in "${PROCS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    RUN_ID="p${p}_sigmoid_h129_bs${batch}_$(date +%Y%m%d_%H%M%S)"
    echo "Running: procs=${p}, act=relu, hidden=228, batch=${batch}"
    mpiexec -n ${p} python main.py \
      --train data/taxi_train.parquet \
      --test  data/taxi_test.parquet \
      --ycol total_amount \
      --act sigmoid --hidden 100 --lr 3e-4 --batch ${batch} \
      --epochs 40 --patience 10 \
      --outdir results/scaling/${RUN_ID} --save-history
    sleep 1
  done
done

```
### 5. Rank Experiment 
  ## Relu
  chmod +x run_scaling_then_best_relu.sh
  
./run_scaling_then_best_relu.sh 2>&1 | tee scaling_then_best_relu.log

  ## sigmoid
  chmod +x run_scaling_then_best_sigmoid.sh

./run_scaling_then_best_sigmoid.sh 2>&1 | tee scaling_then_best_sigmoid.log

  ## tanh

  chmod +x run_scaling_then_best_tanh.sh

./run_scaling_then_best_tanh.sh 2>&1 | tee scaling_then_best_tanh.log



---
### 📊 6.Plotting & Reporting
run the mpi function with different process , repeat the same code with change in process
```bash
python src/report_scaling.py \
  --csv results/scaling/results.csv \
  --outdir output/scaling_report
```
```bash
python src/report_scaling.py \
  --csv results/sweep_run/sweep_merged.csv \
  --outdir output/sweep_report
```
```bash
Generates RMSE line, RMSE scatter, loss curves, and a combined training time plot.
python src/report_plots_per_activation.py \
  --results results/scaling/results.csv \
  --outdir results/scaling
```
This writes images into:
	•	results/scaling/per_activation/*.png
	•	results/scaling/plots_per_activation.md

Open the Markdown in VS Code/GitHub to view plots inline.

📦 Outputs

	•	CSV: results.csv (train/test RMSE, runtime, hyperparams).
	•	Loss CSVs: loss_worldX_batchY.csv.
	•	PNG Charts: RMSE vs processes, loss curves, training time.
	•	Markdown Reports: plots.md and plots_per_activation.md.
	•	Logs: per-rank debug logs.




