# Modeling Approach Summary

## Overview

This project implements a comprehensive machine learning pipeline for binary classification using multiple model architectures with a unified preprocessing and ensemble strategy. The approach combines traditional feature engineering with state-of-the-art text embeddings, addressing class imbalance and optimizing performance through systematic hyperparameter tuning and threshold optimization.

## Feature Engineering

**Regular Features:** 54 numeric features extracted from the raw data, including statistical and domain-specific attributes.

**Embedding Families:** Three pre-trained embedding models were used to capture semantic information:
- **Sentence Transformers** (`all-MiniLM-L6-v2`): 384-dimensional embeddings
- **SciBERT** (`allenai/scibert_scivocab_uncased`): 768-dimensional embeddings  
- **SPECTER2** (`allenai/specter2`): 768-dimensional embeddings

**Feature Combination:** Regular features were concatenated with embeddings, resulting in ~1,974 total features (54 regular + 1,920 embedding dimensions).

## Preprocessing Pipeline

1. **PCA Compression** (for tree-based models): Applied IncrementalPCA to each embedding family, reducing dimensions to 32 components per family to improve computational efficiency while preserving information.

2. **Feature Scaling:** StandardScaler normalization applied to all features to ensure consistent scale across different feature types.

3. **Class Imbalance Handling:** SMOTETomek resampling used to address severe class imbalance (~13:1 negative-to-positive ratio), balancing the dataset to ~2.5:1 ratio.

4. **Memory Management:** Chunked processing and aggressive garbage collection implemented throughout to handle large-scale data efficiently and prevent out-of-memory errors.

## Model Training

**Models Trained:** Five different architectures were trained:
- **XGBoost**: Gradient boosting with tree-based learning
- **LightGBM**: Gradient boosting optimized for speed and efficiency
- **CatBoost**: Gradient boosting with categorical feature handling
- **sklearn MLP**: Multi-layer perceptron neural network
- **PyTorch MLP**: Deep neural network with batch normalization, dropout, and residual connections

**Training Strategy:**
- **Cross-Validation:** 5-fold stratified cross-validation for robust performance estimation
- **Hyperparameter Tuning:** RandomizedSearchCV/GridSearchCV with comprehensive parameter grids
- **Threshold Optimization:** Fine-grained threshold search (120+ thresholds) using precision-recall curves to maximize F1 score
- **Model Calibration:** Isotonic calibration applied to improve probability calibration
- **Early Stopping:** Implemented for gradient boosting models to prevent overfitting

## Ensemble Strategy

**Weighted Ensemble:** Final predictions combine all models using performance-weighted averaging, where weights are proportional to each model's validation F1 score squared.

**Ensemble Methods Available:**
- **Weighted Mean** (default): Performance-weighted average of probabilities
- **Geometric Mean**: Geometric mean of probabilities
- **Majority Voting**: Binary voting with tie-breaking strategies
- **Rank-based**: Average of rank-transformed predictions

**Tie-Breaking:** Conservative strategy (predict negative on ties) to minimize false positives, with options for aggressive or probability-based tie-breaking.

## Key Technical Innovations

- **OOM-Resistant Design:** Chunked processing, memory monitoring, and checkpoint/resume capabilities
- **Unified Pipeline:** Consistent preprocessing across all models ensures fair comparison
- **PyTorch MLP Exception:** Uses full embeddings (no PCA) to leverage neural network capacity
- **Robust Evaluation:** Comprehensive metrics including F1, ROC-AUC, precision-recall curves, and confusion matrices
- **Production-Ready:** Model serialization with all preprocessing components (scalers, PCA models, thresholds) saved together

## Results

The ensemble approach leverages the complementary strengths of different model architectures, with tree-based models capturing feature interactions and neural networks learning complex non-linear patterns in the high-dimensional embedding space. The weighted ensemble strategy ensures that better-performing models contribute more to final predictions while maintaining diversity through different model architectures.

