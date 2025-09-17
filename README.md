# DSA5208_Stochastic_Gradient_Descent_MPI



## ⚙️ Setup

You can set up the environment in **two ways**:  

**Option 1: pip**

python -m venv venv


source venv/bin/activate   # (Linux/Mac)


venv\Scripts\activate      # (Windows)


pip install -r requirements.txt

**Option 2: conda (recommended, ensures MPI works)**

conda env create -f environment.yml

conda activate mpi-sgd-env


## Repository Structure
├── src/
│   ├── prep.py              # data cleaning & split logic
│   └── …
├── data/                    # place datasets here
│   ├── nytaxi2022_cleaned.csv   # cleansed dataset (shared via OneDrive)
│   ├── taxi_test.csv            # test set (shared via OneDrive)
│   ├── taxi_train.csv           # training set (shared via OneDrive)
│   ├── taxi_test.parquet        # test set (shared via OneDrive)
│   ├── taxi_train.parquet       # training set (shared via OneDrive)
├── .gitignore
└── README.md

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


