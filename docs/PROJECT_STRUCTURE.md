# Project Structure

## Folder Organization

```
Kaggle2/
├── docs/              # Documentation files (.md)
├── logs/              # Log files (.log, pipeline_results.json)
├── pids/              # Process ID files (.pid)
├── results/           # Visualization outputs (.png, .pdf, .svg)
├── data/              # Data directory (at root level)
│   ├── raw/           # Input JSONL files (extracted from archive)
│   ├── processed/     # Processed parquet files (converted from JSONL)
│   └── results/       # Exploration and prediction results
├── archive/           # Archived zip files
├── models/            # Transformer models (SPECTER2, SciBERT)
├── src/               # Source code
│   ├── notebooks/     # Jupyter notebooks
│   │   ├── data_exploration.ipynb      # Feature extraction and exploration
│   │   └── citation_prediction.ipynb   # Model training and prediction
│   ├── scripts/       # Python and shell scripts
│   │   ├── convert_to_parquet.py       # JSONL to Parquet converter
│   │   ├── download_models.py         # Download transformer models
│   │   ├── check_pipeline.sh           # Quick pipeline status check
│   │   ├── monitor_pipeline.sh         # Continuous pipeline monitoring
│   │   └── prepare_for_greatlakes.sh   # Prepare deployment package
│   └── pipeline.py    # Main integration pipeline
├── venv/              # Python virtual environment
├── requirements.txt   # Python dependencies
└── README.md          # Project overview and usage
```

## Data Flow

1. **Raw Data**: `data/raw/` - JSONL files extracted from archive
2. **Processed Data**: `data/processed/` - Parquet files (converted from JSONL)
3. **Results**: `data/results/` - Outputs from notebooks:
   - `X_train.parquet`, `X_val.parquet`, `X_test.parquet` - Processed features
   - `y_train.npy`, `y_val.npy` - Labels
   - `feature_info.json` - Feature metadata
   - `test_predictions.npy` - Final predictions

## File Locations

- **Logs**: `logs/pipeline_run.log`, `logs/pipeline_results.json`
- **PID Files**: `pids/pipeline.pid`
- **Documentation**: `docs/README.md`, `docs/PROJECT_STRUCTURE.md`
- **Visualizations**: `results/correlation_matrix.png`, `results/threshold_tuning.png`
- **Data**: 
  - Input: `data/raw/` (JSONL from archive)
  - Processed: `data/processed/` (parquet files)
  - Outputs: `data/results/` (exploration results, predictions)

## Path References

- **Notebooks** read from: `data/processed/` (parquet files) and `data/results/` (pre-processed features)
- **Notebooks** write to: `data/results/` (exploration results, predictions)
- **Pipeline** extracts to: `data/raw/` (JSONL from archive)
- **Pipeline** converts to: `data/processed/` (parquet files)
- **Scripts** location: `src/scripts/` (Python and shell scripts)
