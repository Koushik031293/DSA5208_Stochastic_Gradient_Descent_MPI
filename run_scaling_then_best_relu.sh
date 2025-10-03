#!/bin/bash
set -euo pipefail


# choosing the batch and process

BATCHES=(32 64 128 256 512)
PROCS=(1 2 3 4)

TRAIN="data/taxi_train.parquet"
TEST="data/taxi_test.parquet"
YCOL="total_amount"

ACT="relu"
HIDDEN=228
LR=3e-4

EPOCHS_SWEEP=5        # fast sweep
EPOCHS_FINAL=20       # deep run on best
PATIENCE=10
OUTDIR="results/scaling"
LOGDIR="logs"

PYTHON_BIN=${PYTHON_BIN:-python}
MPI_BIN=${MPI_BIN:-mpiexec}

mkdir -p "${OUTDIR}" "${LOGDIR}"

echo ">>> Starting sweep at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ---------------------------
# Sweep: p=1..4 × batches (epochs=5)
# ---------------------------
for p in "${PROCS[@]}"; do
  for b in "${BATCHES[@]}"; do
    echo "========================================="
    echo "Running: procs=${p}, batch=${b}, epochs=${EPOCHS_SWEEP}"
    echo "========================================="

    LOGFILE="${LOGDIR}/run_${ACT}_world${p}_batch${b}.log"

    ${MPI_BIN} -n ${p} \
      -x PYTHONUNBUFFERED=1 -x MPLBACKEND=Agg \
      ${PYTHON_BIN} main.py \
        --train "${TRAIN}" \
        --test  "${TEST}" \
        --ycol  "${YCOL}" \
        --act "${ACT}" --hidden ${HIDDEN} --lr ${LR} \
        --batch ${b} \
        --epochs ${EPOCHS_SWEEP} --patience ${PATIENCE} \
        --outdir "${OUTDIR}" --save-history \
        --history-filename "loss_${ACT}_world${p}_batch${b}" \
      2>&1 | tee "${LOGFILE}"

    echo "✓ Log saved to ${LOGFILE}"
    echo ""
  done
done

RESULTS_CSV="${OUTDIR}/results.csv"
if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "ERROR: ${RESULTS_CSV} not found. Did runs complete?"
  exit 2
fi

echo ">>> Selecting best config from the last sweep rows …"

RESULTS_CSV="${OUTDIR}/results.csv"
if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "ERROR: ${RESULTS_CSV} not found. Did runs complete?"
  exit 2
fi

# How many entries should this sweep add?
NUM_PROCS=${#PROCS[@]}
NUM_BATCHES=${#BATCHES[@]}
EXPECTED_ROWS=$(( NUM_PROCS * NUM_BATCHES ))

# How many rows are actually in the CSV (excluding header)?
TOTAL_LINES=$(wc -l < "${RESULTS_CSV}")
DATA_LINES=$(( TOTAL_LINES - 1 ))
if (( DATA_LINES <= 0 )); then
  echo "ERROR: ${RESULTS_CSV} has no data rows yet."
  echo "Tail of file:"
  tail -n 5 "${RESULTS_CSV}" || true
  exit 3
fi

# If fewer than EXPECTED_ROWS exist, just take what we have
TAKE_ROWS=$(( DATA_LINES < EXPECTED_ROWS ? DATA_LINES : EXPECTED_ROWS ))

echo ">>> CSV has ${DATA_LINES} data rows; taking last ${TAKE_ROWS} for selection (expected ${EXPECTED_ROWS})."

TAIL_CSV="${OUTDIR}/results_tail_for_selection.csv"
{ head -n 1 "${RESULTS_CSV}"; tail -n "${TAKE_ROWS}" "${RESULTS_CSV}"; } > "${TAIL_CSV}"

# Select best: first try strict (match activation/hidden + allowed worlds/batches).
# If empty, relax step-by-step and print why.
BEST_JSON=$(cat << 'PYCODE' | ${PYTHON_BIN} /dev/stdin "${TAIL_CSV}" "$(IFS=,; echo "${PROCS[*]}")" "$(IFS=,; echo "${BATCHES[*]}")" "${ACT}" "${HIDDEN}"
import csv, json, sys

results_csv, procs_str, batches_str, act_want, hidden_want = sys.argv[1:6]
allowed_worlds = {int(x) for x in procs_str.split(",") if x}
allowed_batches = {int(x) for x in batches_str.split(",") if x}

def load_rows(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print("DEBUG: failed to read CSV:", e)
        return []

rows = load_rows(results_csv)
if not rows:
    print("{}"); sys.exit(0)

def to_num(row, key, typ=float, default=None):
    try:
        return typ(row[key])
    except Exception:
        return default

# Try strict filter first
def pick(rows, strict=True):
    cand = []
    for r in rows:
        w = to_num(r, "world", int)
        b = to_num(r, "batch", int)
        tr = to_num(r, "train_rmse", float)
        te = to_num(r, "test_rmse", float)
        tt = to_num(r, "train_time_s", float)
        if w is None or b is None or tr is None or te is None or tt is None:
            continue
        if strict:
            if r.get("activation") != act_want: continue
            if str(r.get("hidden")) != str(hidden_want): continue
        if w not in allowed_worlds or b not in allowed_batches:
            continue
        cand.append({"world": w, "batch": b, "train_rmse": tr, "test_rmse": te, "train_time_s": tt, "lr": r.get("lr","")})
    cand.sort(key=lambda x: (x["test_rmse"], x["train_time_s"]))
    return cand

strict = pick(rows, strict=True)
if strict:
    print(json.dumps(strict[0])); sys.exit(0)

# Relax activation/hidden match
relaxed = pick(rows, strict=False)
if relaxed:
    print(json.dumps(relaxed[0])); sys.exit(0)

# Nothing matched even relaxed: return {} and let shell print a helpful error
print("{}")
PYCODE
)

if [[ "${BEST_JSON}" == "{}" || -z "${BEST_JSON}" ]]; then
  echo "ERROR: Could not find any valid rows among the last ${TAKE_ROWS} entries of ${RESULTS_CSV}."
  echo "DEBUG: showing the last ${TAKE_ROWS} rows:"
  tail -n "${TAKE_ROWS}" "${RESULTS_CSV}" | sed -n '1,10p'
  echo "HINTS:"
  echo "  • Verify runs are appending to OUTDIR=${OUTDIR}"
  echo "  • Confirm results.csv has columns: world,batch,train_rmse,test_rmse,train_time_s"
  echo "  • Ensure your run actually completed (each rank exits cleanly) so _save_results runs on rank 0."
  exit 3
fi

WORLD=$(echo "${BEST_JSON}" | ${PYTHON_BIN} -c 'import sys, json; print(json.load(sys.stdin)["world"])')
BATCH=$(echo "${BEST_JSON}" | ${PYTHON_BIN} -c 'import sys, json; print(json.load(sys.stdin)["batch"])')
LR_EXACT=$(echo "${BEST_JSON}" | ${PYTHON_BIN} -c 'import sys, json; print(json.load(sys.stdin)["lr"])')

echo ">>> Best config (from last ${TAKE_ROWS} rows): world=${WORLD}, batch=${BATCH}, lr=${LR_EXACT}"
echo ">>> Re-running best config with epochs=${EPOCHS_FINAL} …"

FINAL_LOGFILE="${LOGDIR}/run_${ACT}_world${WORLD}_batch${BATCH}_final.log"

${MPI_BIN} -n ${WORLD} \
  -x PYTHONUNBUFFERED=1 -x MPLBACKEND=Agg \
  ${PYTHON_BIN} main.py \
    --train "${TRAIN}" \
    --test  "${TEST}" \
    --ycol  "${YCOL}" \
    --act "${ACT}" --hidden ${HIDDEN} --lr ${LR_EXACT} \
    --batch ${BATCH} \
    --epochs ${EPOCHS_FINAL} --patience ${PATIENCE} \
    --outdir "${OUTDIR}" --save-history \
    --history-filename "loss_${ACT}_world${WORLD}_batch${BATCH}_final" \
  2>&1 | tee "${FINAL_LOGFILE}"

echo "✓ Final log saved to ${FINAL_LOGFILE}"
echo ">>> Done."