#!/bin/bash

#SBATCH --job-name=data_exploration_organized
#SBATCH --account=eecs442f25_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=santoshd@umich.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT,NODE_FAIL

set -euo pipefail
set -o errtrace
umask 077

# Suppress Python warnings
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning"

# ============================================================================
# Configuration and Setup
# ============================================================================

module purge
module load python3.11-anaconda/2024.02
module load cuda/12.1 || true

# Directory setup
mkdir -p logs .pip-cache
export PIP_CACHE_DIR="$PWD/.pip-cache"
export WORK_DIR="${SLURM_TMPDIR:-$PWD}"
export ORIG_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export VENV_DIR="$ORIG_DIR/venv"

# ============================================================================
# Logging Functions
# ============================================================================

log() {
    echo "$@" >&1
    echo "$@" >&2
    sync 2>/dev/null || true
}

# ============================================================================
# Virtual Environment Setup
# ============================================================================

log "Activating virtual environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    log "✗ ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"
export VIRTUAL_ENV_DISABLE_PROMPT=1

mkdir -p "$WORK_DIR/runs" "$WORK_DIR/logs"
ln -snf "$WORK_DIR/runs" runs 2>/dev/null || true

# ============================================================================
# Environment Variables for Performance
# ============================================================================

export PYTORCH_ALLOC_CONF="expandable_segments:true,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ============================================================================
# System Information
# ============================================================================

set +u
log "=========================================="
log "JOB STARTUP INFORMATION"
log "=========================================="
log "Host:        $(hostname)"
log "Date:        $(date -Is)"
log "SLURM_JOBID: ${SLURM_JOB_ID:-none}"
log "Working directory: $(pwd)"
log "Python:      $(which python 2>/dev/null || echo 'not found')"
log "=========================================="
set -u

# ============================================================================
# Verify Environment
# ============================================================================

log "Verifying Python environment..."
if python -c "import sys; import papermill; import ipykernel; print('✓ Basic packages available', flush=True)" 2>/dev/null; then
    log "✓ Python environment verified"
else
    log "⚠ WARNING: Could not verify all packages"
fi

# ============================================================================
# Jupyter Kernel Setup
# ============================================================================

KNAME="data-exploration-organized-${SLURM_JOB_ID:-$$}"
log "Installing Jupyter kernel: ${KNAME}..."
if python -m ipykernel install --user --name "${KNAME}" --display-name "Data Exploration Organized (${KNAME})" 2>&1; then
    log "✓ Kernel installed successfully"
    trap 'jupyter kernelspec remove -y "${KNAME}" 2>/dev/null || true' EXIT
fi

# ============================================================================
# Notebook Configuration
# ============================================================================

NOTEBOOK_IN="src/notebooks/data_exploration_organized.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
NOTEBOOK_OUT="$WORK_DIR/runs/data_exploration_organized_executed_${STAMP}.ipynb"

# ============================================================================
# Pre-flight Validation
# ============================================================================

log "Running pre-flight validation..."
VALIDATION_ERRORS=0

# Check for notebook
if [ ! -f "$ORIG_DIR/$NOTEBOOK_IN" ]; then
    log "✗ ERROR: Notebook not found: $NOTEBOOK_IN"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
else
    log "✓ Notebook found: $NOTEBOOK_IN"
fi

if [ $VALIDATION_ERRORS -gt 0 ]; then
    log "❌ PRE-FLIGHT VALIDATION FAILED: $VALIDATION_ERRORS error(s) found"
    exit 1
fi

log "✅ Pre-flight validation passed!"

# ============================================================================
# File Copying (if using SLURM_TMPDIR)
# ============================================================================

if [ -n "${SLURM_TMPDIR:-}" ] && [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    log "Using SLURM_TMPDIR: $SLURM_TMPDIR"
    mkdir -p "$WORK_DIR/src/notebooks"
    
    cp -v "$ORIG_DIR/$NOTEBOOK_IN" "$WORK_DIR/$NOTEBOOK_IN" 2>/dev/null || exit 1
    cd "$WORK_DIR"
fi

# ============================================================================
# Cleanup Previous Instances
# ============================================================================

log "=== Cleaning Previous Runs ==="
# Kill any previous instances
pkill -f "papermill.*data_exploration_organized" 2>/dev/null || true
pkill -f "ipykernel.*data-exploration-organized" 2>/dev/null || true
# Clear previous logs
rm -f "$WORK_DIR/logs/data_exploration_organized_run.log" 2>/dev/null || true
sleep 2
log "✓ Cleanup completed"

# ============================================================================
# Notebook Execution
# ============================================================================

log "Running papermill -> ${NOTEBOOK_OUT}"
log "Notebook: ${NOTEBOOK_IN}"
log "Kernel: ${KNAME}"
NOTEBOOK_START=$(date +%s)

# Run with proper logging - capture both stdout and stderr
LOG_FILE="$WORK_DIR/logs/data_exploration_organized_run.log"
mkdir -p "$(dirname "$LOG_FILE")"

if ! papermill "${NOTEBOOK_IN}" "${NOTEBOOK_OUT}" \
    --kernel-name "${KNAME}" \
    --log-output \
    --execution-timeout 14400 \
    2>&1 | tee "$LOG_FILE"; then
    NOTEBOOK_END=$(date +%s)
    NOTEBOOK_DURATION=$((NOTEBOOK_END - NOTEBOOK_START))
    log "✗ ERROR: Papermill failed after ${NOTEBOOK_DURATION}s!"
    log "Check log file: $LOG_FILE"
    exit 1
fi

NOTEBOOK_END=$(date +%s)
NOTEBOOK_DURATION=$((NOTEBOOK_END - NOTEBOOK_START))
log "✓ Notebook execution completed in ${NOTEBOOK_DURATION}s"

# ============================================================================
# Copy Results Back
# ============================================================================

if [ -n "${SLURM_TMPDIR:-}" ] && [ "$WORK_DIR" != "$ORIG_DIR" ]; then
    log "Copying results back..."
    
    # Copy logs
    if [ -f "$LOG_FILE" ]; then
        mkdir -p "$ORIG_DIR/logs"
        cp -v "$LOG_FILE" "$ORIG_DIR/logs/" 2>/dev/null || true
    fi
    
    # Copy executed notebook
    if [ -n "${SLURM_JOB_ID:-}" ] && [ -f "${NOTEBOOK_OUT}" ]; then
        RUN_OUT_DIR="$ORIG_DIR/runs/run_${SLURM_JOB_ID}"
        mkdir -p "$RUN_OUT_DIR"
        cp -v "${NOTEBOOK_OUT}" "$RUN_OUT_DIR/" 2>/dev/null || true
    fi
fi

log ""
log "============================================================"
log "EXECUTION SUMMARY"
log "============================================================"
log "Notebook execution time: ${NOTEBOOK_DURATION}s"
log "============================================================"

