# 📘 DSA5208 – Stochastic Gradient Descent with MPI

## 📌 Project Overview
This project implements a **parallel Stochastic Gradient Descent (SGD)** training loop using **MPI (Message Passing Interface)** via the `mpi4py` Python library.  
The aim is to train a simple feed-forward neural network on large tabular datasets by **distributing computation across processes**.  

- **Language:** Python 3  
- **Parallelism:** MPI (`mpi4py`)  
- **Dataset format:** Parquet (`.parquet`)  

---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourname/DSA5208_MPI_SGD.git
   cd DSA5208_MPI_SGD
   ```

2. Create environment & install dependencies:
   ```bash
   conda create -n dsa5208 python=3.10 -y
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
│   ├── prep.py           # Data preprocessing / split
│   ├── train_mpi_sgd.py  # SGD training with MPI
│   ├── main.py           # Main entry point (handles args, runs training)
│   └── utils.py          # Helper functions
├── output/               # Results, logs, plots, metrics
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🚀 Running the Code

### 1. Preprocess the dataset
```bash
python src/prep.py --input raw_data.csv --output data/
```

### 2. Train with MPI
Run with **4 processes** on the taxi dataset:
```bash
mpirun -n 4 python src/main.py     --train data/taxi_train.parquet     --test data/taxi_test.parquet     --ycol total_amount     --epochs 20     --lr 0.001     --batch 128     --act relu
```

### 3. Sweep Experiments
Grid search over activations and batch sizes:
```bash
mpirun -n 4 python src/main.py     --sweep     --train data/taxi_train.parquet     --test data/taxi_test.parquet     --ycol total_amount
```

---

## 📊 Outputs
- **Metrics CSV** (train/test RMSE, runtime)  
- **Plots** (loss curves, comparison charts)  
- **Logs** (per-rank progress, debug info)  

---


## 📌 Requirements
- Python ≥ 3.9  
- mpi4py ≥ 3.1  
- pandas, numpy, matplotlib  

Install with:
```bash
pip install -r requirements.txt
```

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