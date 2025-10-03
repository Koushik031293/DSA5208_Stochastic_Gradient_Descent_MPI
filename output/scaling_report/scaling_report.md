# MPI-SGD Scaling Report

_Generated: 2025-09-27 07:17:17Z_

**Input CSV:** `/Users/koushik/Desktop/MS/DSA5208/Project/DSA5208_Stochastic_Gradient_Descent_MPI/results/scaling/results.csv`

## RMSE of Training and Test Data
- Rows: 5
- **Best Test RMSE**: 3.176940 (train=3.163220, world=1)
- Full table saved to `rmse_table.csv`.

## Training Times for Different Numbers of Processes
- Summary saved to `time_summary_by_world.csv`.
- Speedup/Efficiency saved to `speedup_vs_1.csv`.

## Efforts Made to Improve Results
We applied the following improvements during experimentation:
1. **Feature standardization & numeric sanitation:** replaced NaN/inf, standardized using train-set statistics to stabilize gradients.
2. **Weight initialization (Glorot):** keeps activations/gradients in reasonable ranges for shallow MLPs.
3. **Activation selection:** preferring `tanh` for the one-hidden-layer network to reduce dead units and improve convergence stability under synchronous MPI gradient averaging.
4. **Mini-batch size tuning:** swept batch sizes {32, 64, 128, 256, 512}; selected 128–256 as a good trade-off between generalization and wall-clock time.
5. **Learning-rate tuning & early stopping:** explored lr in {1e-3, 3e-4}; enabled patience-based early stopping to avoid overfitting and wasted epochs.
6. **Synchronous gradient averaging:** used MPI Allreduce to average gradients across ranks each step, improving stability vs. naive parameter push.
7. **Per-epoch history logging:** recorded `loss_curve.csv` to inspect R(θ_k) vs epoch; used this to verify steady descent and detect plateaus.
8. **Process scaling runs (1→4):** measured train time per world size, computed speedup & efficiency to justify the chosen parallel configuration.
