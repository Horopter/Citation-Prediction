# Notebook Workflow and Purpose

This document explains the purpose and flow of each notebook in the project.

## Overview: Complete Pipeline Flow

```
Raw JSONL Files (data/raw/)
    ↓
[1] data_exploration_organized.ipynb
    ↓
Intermediate Features & Embeddings (data/results/)
    ↓
[2] data_exploration_next_steps.ipynb
    ↓
Model-Ready Data (data/model_ready/)
    ↓
[3] Model Training Notebooks (XGBoost, LightGBM, CatBoost, PyTorch MLP)
    ↓
Trained Models (models/saved_models/)
    ↓
[4] ensemble_predict.py
    ↓
Final Predictions (data/submission_files/)
```

---

## 1. Data Exploration Notebooks

### `data_exploration_organized.ipynb`
**Purpose**: Extract features and generate embeddings from raw JSONL files

**What it does**:
- **Input**: Raw JSONL files (`train.jsonl`, `val.jsonl`, `test.no_label.jsonl`) from `data/raw/`
- **Feature Extraction**: Extracts 70+ structured features from nested JSON:
  - Metadata features (publication year, type, open access status)
  - Author statistics (count, affiliation info)
  - Concept features (count, relevance scores)
  - Text statistics (abstract length, word counts)
  - NLP features (regex-based patterns)
- **Embedding Generation**: Generates text embeddings using:
  - Sentence Transformers (`all-MiniLM-L6-v2`) - 384 dimensions
  - SciBERT (`allenai/scibert_scivocab_uncased`) - 768 dimensions
  - SPECTER2 (`allenai/specter2`) - 768 dimensions
- **Processing**: Chunked processing (default 1000 records) with checkpoint support
- **Output**: Saves to `data/results/`:
  - `X_train.parquet`, `X_val.parquet`, `X_test.parquet` - Feature matrices
  - `y_train.npy`, `y_val.npy` - Labels
  - `sent_transformer_X_*.parquet` - Sentence transformer embeddings
  - `scibert_X_*.parquet` - SciBERT embeddings
  - `specter2_X_*.parquet` - SPECTER2 embeddings

**Key Features**:
- Memory-efficient chunked processing
- Checkpoint system for resumability
- Batch processing for embeddings
- OOM-resistant design

---

### `data_exploration_next_steps.ipynb`
**Purpose**: Combine features and embeddings, perform statistical analysis, create model-ready datasets

**What it does**:
- **Input**: Feature matrices and embeddings from `data/results/`
- **Merging**: Combines base features with all embedding families by `id`
- **Missing Value Handling**: Train-centric imputation strategy
- **Feature Engineering**: 
  - Replaces `is_oa` with `is_not_oa` (1 - is_oa) for better signal
  - Keeps all non-embedding features (no reduction)
- **Statistical Analysis**:
  - Pearson correlation
  - Spearman correlation
  - Chi-square tests
  - Cramér's V
  - ANOVA
  - Tukey's HSD
- **Output**: Saves to `data/model_ready/`:
  - `train_model_ready.parquet` - Combined features ready for modeling
  - `val_model_ready.parquet`
  - `test_model_ready.parquet`
  - Analysis reports (CSV files with correlations, ANOVA results, etc.)

**Key Features**:
- Polars-only (no pandas) for scalability
- Comprehensive statistical analysis
- Preserves all features for downstream modeling

---

## 2. Model Training Notebooks

All model notebooks follow a similar structure but use different algorithms:

### `model_xgboost_all_features.ipynb`
**Purpose**: Train XGBoost classifier with all features

**What it does**:
1. **Load Data**: Reads `train_model_ready.parquet` and `val_model_ready.parquet`
2. **Feature Splitting**: Separates regular features (54) from embeddings (1920 dims)
3. **PCA Compression**: Applies IncrementalPCA to embeddings (32 components per family)
4. **SMOTE Resampling**: Incremental SMOTEENN/SMOTETomek with 10k chunks (OOM-resistant)
5. **Feature Scaling**: StandardScaler normalization
6. **Hyperparameter Tuning**: 5-fold CV with RandomizedSearchCV
7. **Threshold Optimization**: Fine-grained search (120+ thresholds) for optimal F1
8. **Model Calibration**: Isotonic calibration for better probability estimates
9. **Save Model**: Saves to `models/saved_models/model_xgboost_all_features_best.pkl`
10. **Generate Predictions**: Creates submission file

**Key Features**:
- OOM-resistant incremental SMOTE (10k chunks, ENN default)
- Never skips SMOTE (always applies resampling)
- Retry logic for memory errors
- Comprehensive hyperparameter tuning

---

### `model_lightgbm_all_features.ipynb`
**Purpose**: Train LightGBM classifier with all features

**What it does**: Same workflow as XGBoost but uses LightGBM algorithm
- Optimized for speed and efficiency
- Similar preprocessing pipeline
- Different hyperparameter grid

---

### `model_catboost_all_features.ipynb`
**Purpose**: Train CatBoost classifier with all features

**What it does**: Same workflow as XGBoost but uses CatBoost algorithm
- Handles categorical features natively
- Similar preprocessing pipeline
- Different hyperparameter grid

---

### `model_pytorch_mlp_all_features.ipynb`
**Purpose**: Train PyTorch MLP neural network with all features

**What it does**:
1. **Load Data**: Reads model-ready parquet files
2. **Feature Splitting**: Separates regular features from embeddings
3. **NO PCA**: Uses full embeddings (no compression) - leverages neural network capacity
4. **Class Imbalance**: Handled via `pos_weight_tensor` in loss function (no SMOTE for speed)
5. **Feature Scaling**: StandardScaler normalization
6. **Training**: Fixed hyperparameters with CV validation
7. **Threshold Optimization**: Fine-grained search
8. **Save Model**: Saves PyTorch state dict and config
9. **Generate Predictions**: Creates submission file

**Key Features**:
- Uses full embeddings (no PCA)
- GPU acceleration with CPU fallback
- Class imbalance via loss function weights
- Deep neural network with batch normalization and dropout

---

## 3. Ensemble Prediction

### `ensemble_predict.py` (Script, not notebook)
**Purpose**: Combine predictions from all trained models using weighted ensemble

**What it does**:
1. **Load Models**: Loads all saved models from `models/saved_models/`
2. **Load Test Data**: Reads `test_model_ready.parquet`
3. **Preprocess**: Applies same preprocessing as training (PCA, scaling)
4. **Get Predictions**: Gets probabilities from each model
5. **Ensemble**: Combines using weighted mean (default) or other methods:
   - Weighted mean (performance-weighted)
   - Geometric mean
   - Majority voting
   - Rank-based averaging
6. **Threshold**: Applies optimal threshold (average of model thresholds)
7. **Save**: Creates `submission_ensemble.csv`

**Key Features**:
- Supports both sklearn and PyTorch models
- Chunked processing for large datasets
- Multiple ensemble methods
- Tie-breaking strategies

---

## Complete Workflow Summary

### Phase 1: Data Preparation
1. **Convert JSONL to Parquet** (`src/scripts/convert_to_parquet.py`)
   - Converts raw JSONL files to parquet format
   - Output: `data/processed/*.parquet`

2. **Feature Extraction** (`data_exploration_organized.ipynb`)
   - Extracts structured features
   - Generates text embeddings
   - Output: `data/results/X_*.parquet`, `y_*.npy`, embedding files

3. **Feature Engineering** (`data_exploration_next_steps.ipynb`)
   - Combines features and embeddings
   - Statistical analysis
   - Missing value imputation
   - Output: `data/model_ready/*_model_ready.parquet`

### Phase 2: Model Training
4. **Train Individual Models** (One of: `model_xgboost_*.ipynb`, `model_lightgbm_*.ipynb`, `model_catboost_*.ipynb`, `model_pytorch_mlp_*.ipynb`)
   - Each notebook trains one model architecture
   - Hyperparameter tuning
   - Threshold optimization
   - Output: `models/saved_models/model_*_all_features_best.pkl`

### Phase 3: Ensemble & Prediction
5. **Ensemble Prediction** (`ensemble_predict.py`)
   - Combines all model predictions
   - Generates final submission
   - Output: `data/submission_files/submission_ensemble.csv`

---

## Data Flow Diagram

```
data/raw/
├── train.jsonl ──────────────┐
├── val.jsonl ────────────────┤
└── test.no_label.jsonl ──────┘
         │
         │ [convert_to_parquet.py]
         ↓
data/processed/
├── train.parquet ────────────┐
├── val.parquet ──────────────┤
└── test.parquet ──────────────┘
         │
         │ [data_exploration_organized.ipynb]
         ↓
data/results/
├── X_train.parquet ──────────┐
├── X_val.parquet ─────────────┤
├── X_test.parquet ────────────┤
├── y_train.npy ───────────────┤
├── y_val.npy ──────────────────┤
├── sent_transformer_X_*.parquet│
├── scibert_X_*.parquet ────────┤
└── specter2_X_*.parquet ───────┘
         │
         │ [data_exploration_next_steps.ipynb]
         ↓
data/model_ready/
├── train_model_ready.parquet ─┐
├── val_model_ready.parquet ───┤
└── test_model_ready.parquet ──┘
         │
         │ [model_*_all_features.ipynb]
         ↓
models/saved_models/
├── model_xgboost_*.pkl ────────┐
├── model_lightgbm_*.pkl ───────┤
├── model_catboost_*.pkl ───────┤
└── model_pytorch_mlp_*.pkl ─────┘
         │
         │ [ensemble_predict.py]
         ↓
data/submission_files/
└── submission_ensemble.csv
```

---

## Key Differences Between Model Notebooks

| Feature | XGBoost | LightGBM | CatBoost | PyTorch MLP |
|---------|---------|----------|----------|-------------|
| **PCA** | Yes (32 comp/family) | Yes (32 comp/family) | Yes (32 comp/family) | No (full embeddings) |
| **SMOTE** | Incremental ENN/Tomek | SMOTETomek | SMOTETomek | None (pos_weight) |
| **Hyperparameter Tuning** | RandomizedSearchCV | RandomizedSearchCV | RandomizedSearchCV | Fixed params + CV |
| **Chunk Size** | 10k (SMOTE) | N/A | N/A | N/A |
| **GPU Support** | CPU only | CPU only | CPU only | GPU + CPU |

---

## Execution Order

1. **First Time Setup**:
   ```bash
   # 1. Convert JSONL to Parquet
   python3 src/scripts/convert_to_parquet.py
   
   # 2. Extract features and embeddings
   jupyter notebook src/notebooks/data_exploration_organized.ipynb
   
   # 3. Create model-ready datasets
   jupyter notebook src/notebooks/data_exploration_next_steps.ipynb
   ```

2. **Train Models** (can run in parallel):
   ```bash
   # Train each model (run separately or in parallel)
   jupyter notebook src/notebooks/model_xgboost_all_features.ipynb
   jupyter notebook src/notebooks/model_lightgbm_all_features.ipynb
   jupyter notebook src/notebooks/model_catboost_all_features.ipynb
   jupyter notebook src/notebooks/model_pytorch_mlp_all_features.ipynb
   ```

3. **Generate Ensemble Predictions**:
   ```bash
   python3 ensemble_predict.py
   ```

---

## Notes

- **Data Exploration notebooks** are typically run once to prepare the data
- **Model Training notebooks** can be run independently and in parallel
- **Ensemble script** requires all models to be trained first
- All notebooks use Polars for dataframes (no pandas) for better memory efficiency
- All notebooks include OOM-resistant design with chunked processing

