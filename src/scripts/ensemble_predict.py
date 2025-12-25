#!/usr/bin/env python3
"""
Scalable ensemble prediction script that handles any number of models with proper tie-breaking.
Supports both sklearn models (XGBoost, LightGBM, CatBoost) and PyTorch MLP models.
"""

import pickle
import polars as pl
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings
import re
warnings.filterwarnings('ignore')

# Suppress warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Try importing torch (for PyTorch MLP)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. PyTorch MLP models will be skipped.")


def extract_work_id(id_value: str) -> str:
    """Extract work_id from URL or return as is if already just ID."""
    id_str = str(id_value)
    if id_str.startswith('W') and len(id_str) > 1 and '/' not in id_str:
        return id_str
    match = re.search(r'W\d+', id_str)
    if match:
        return match.group(0)
    return id_str


def load_saved_model(model_path: Path) -> Dict[str, Any]:
    """Load a saved model pickle file."""
    print(f"📦 Loading model: {model_path.name}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def load_test_data(test_path: Path) -> pl.DataFrame:
    """Load test data from parquet file."""
    print(f"📊 Loading test data: {test_path.name}")
    df = pl.read_parquet(test_path)
    print(f"   Shape: {df.shape}")
    return df


def prepare_features_sklearn(df: pl.DataFrame, model_data: Dict[str, Any]) -> np.ndarray:
    """Prepare features for sklearn models (with PCA)."""
    reg_cols = model_data.get('reg_cols', [])
    emb_family_to_cols = model_data.get('emb_family_to_cols', {})
    pca_models = model_data.get('pca_models', {})
    
    # Extract regular features
    X_reg = None
    if reg_cols:
        available_reg_cols = [col for col in reg_cols if col in df.columns]
        if available_reg_cols:
            X_reg = df.select(available_reg_cols).to_numpy()
    
    # Extract and apply PCA to embeddings
    X_emb_pca_list = []
    if pca_models and emb_family_to_cols:
        for emb_family, emb_cols in emb_family_to_cols.items():
            if not emb_cols:
                continue
            
            available_emb_cols = [col for col in emb_cols if col in df.columns]
            if not available_emb_cols:
                continue
            
            X_emb = df.select(available_emb_cols).to_numpy()
            
            if emb_family in pca_models and pca_models[emb_family] is not None:
                pca_model = pca_models[emb_family]
                X_emb_pca = pca_model.transform(X_emb)
                X_emb_pca_list.append(X_emb_pca)
            else:
                X_emb_pca_list.append(X_emb)
    
    # Combine features
    feature_parts = []
    if X_reg is not None:
        feature_parts.append(X_reg)
    if X_emb_pca_list:
        feature_parts.extend(X_emb_pca_list)
    
    if not feature_parts:
        raise ValueError("No features extracted!")
    
    X = np.hstack(feature_parts) if len(feature_parts) > 1 else feature_parts[0]
    
    # Apply scaling
    scaler = model_data.get('scaler')
    if scaler is not None:
        expected_features = scaler.n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Feature mismatch: expected {expected_features}, got {X.shape[1]}"
            )
        X = scaler.transform(X)
    
    return X


def prepare_features_pytorch(df: pl.DataFrame, model_data: Dict[str, Any]) -> np.ndarray:
    """Prepare features for PyTorch MLP (no PCA)."""
    reg_cols = model_data.get('reg_cols', [])
    emb_family_to_cols = model_data.get('emb_family_to_cols', {})
    
    # Extract regular features
    X_reg = None
    if reg_cols:
        available_reg_cols = [col for col in reg_cols if col in df.columns]
        if available_reg_cols:
            X_reg = df.select(available_reg_cols).to_numpy()
    
    # Extract ALL embeddings (NO PCA)
    X_emb_list = []
    if emb_family_to_cols:
        for emb_family, emb_cols in emb_family_to_cols.items():
            if not emb_cols:
                continue
            
            available_emb_cols = [col for col in emb_cols if col in df.columns]
            if not available_emb_cols:
                continue
            
            X_emb = df.select(available_emb_cols).to_numpy()
            X_emb_list.append(X_emb)
    
    # Combine features
    feature_parts = []
    if X_reg is not None:
        feature_parts.append(X_reg)
    if X_emb_list:
        feature_parts.extend(X_emb_list)
    
    if not feature_parts:
        raise ValueError("No features extracted!")
    
    X = np.hstack(feature_parts) if len(feature_parts) > 1 else feature_parts[0]
    
    # Apply scaling
    scaler = model_data.get('scaler')
    if scaler is not None:
        expected_features = scaler.n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Feature mismatch: expected {expected_features}, got {X.shape[1]}"
            )
        X = scaler.transform(X)
    
    return X


def predict_sklearn_model(model_data: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Make predictions using sklearn-compatible models."""
    model = model_data.get('model')
    if model is None:
        raise ValueError("Model not found!")
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X).astype(float)
    
    return y_proba


def predict_pytorch_model(model_data: Dict[str, Any], X: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Make predictions using PyTorch MLP model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available!")
    
    model_config = model_data.get('model_config', {})
    model_state_dict = model_data.get('model_state_dict')
    
    if model_state_dict is None or not model_config:
        raise ValueError("PyTorch model state dict or config not found!")
    
    # Define MLP class (must match training)
    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: tuple, dropout_rate: float = 0.0,
                     activation: str = 'relu', use_batch_norm: bool = True, use_residual: bool = False):
            super(MLP, self).__init__()
            self.use_residual = use_residual
            self.layers = nn.ModuleList()
            prev_dim = input_dim
            
            for i, hidden_dim in enumerate(hidden_dims):
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
                
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'gelu':
                    self.layers.append(nn.GELU())
                elif activation == 'swish':
                    self.layers.append(nn.SiLU())
                else:
                    self.layers.append(nn.ReLU())
                
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
                
                prev_dim = hidden_dim
            
            self.output = nn.Sequential(
                nn.Linear(prev_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            if self.use_residual and len(self.layers) > 0:
                residual = x
            
            for layer in self.layers:
                x = layer(x)
            
            if self.use_residual and x.shape == residual.shape:
                x = x + residual
            
            return self.output(x).squeeze()
    
    # Create and load model
    model = MLP(
        input_dim=model_config['input_dim'],
        hidden_dims=model_config['hidden_dims'],
        dropout_rate=model_config.get('dropout_rate', 0.3),
        activation=model_config.get('activation', 'swish'),
        use_batch_norm=model_config.get('use_batch_norm', True),
        use_residual=model_config.get('use_residual', False)
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    model = model.to(device)
    
    # Predict in chunks
    chunk_size = 10000
    y_proba_list = []
    
    with torch.no_grad():
        for i in range(0, len(X), chunk_size):
            X_chunk = torch.FloatTensor(X[i:i + chunk_size]).to(device)
            y_chunk = model(X_chunk).cpu().numpy()
            y_proba_list.append(y_chunk)
    
    return np.concatenate(y_proba_list)


def get_model_weight(model_data: Dict[str, Any], default_weight: float = 1.0) -> float:
    """Get weight for a model based on its performance (CV F1 score)."""
    best_f1 = model_data.get('best_f1')
    best_cv_score = model_data.get('best_cv_score')
    
    # Use best_f1 if available, otherwise best_cv_score
    score = best_f1 if best_f1 is not None else best_cv_score
    
    if score is None or score <= 0:
        return default_weight
    
    # Weight proportional to performance (squared to emphasize better models)
    weight = score ** 2
    return weight


def ensemble_predict(
    model_predictions: List[Tuple[np.ndarray, Dict[str, Any]]],
    method: str = 'weighted_mean',
    tie_breaker: str = 'conservative'
) -> np.ndarray:
    """
    Make ensemble predictions from multiple models with proper tie-breaking.
    
    Args:
        model_predictions: List of (predictions, model_data) tuples
        method: Ensemble method:
            - 'mean': Simple average
            - 'weighted_mean': Weighted average by model performance
            - 'geometric_mean': Geometric mean of probabilities
            - 'rank_mean': Average of rank-transformed predictions
            - 'vote': Majority voting with tie-breaking
        tie_breaker: For voting with ties:
            - 'conservative': Predict negative (0) on ties
            - 'aggressive': Predict positive (1) on ties
            - 'prob_mean': Use mean probability to break ties
    
    Returns:
        Ensemble probabilities
    """
    if not model_predictions:
        raise ValueError("No model predictions provided!")
    
    n_models = len(model_predictions)
    predictions = np.array([pred for pred, _ in model_predictions])
    
    print(f"   Ensemble method: {method}")
    print(f"   Number of models: {n_models}")
    
    if method == 'mean':
        ensemble_proba = np.mean(predictions, axis=0)
        
    elif method == 'weighted_mean':
        weights = np.array([get_model_weight(md) for _, md in model_predictions])
        weights = weights / weights.sum()  # Normalize
        ensemble_proba = np.average(predictions, axis=0, weights=weights)
        print(f"   Model weights: {dict(zip([md.get('model_name', f'Model_{i}') for i, (_, md) in enumerate(model_predictions)], weights.round(3)))}")
        
    elif method == 'geometric_mean':
        # Geometric mean: (prod)^(1/n)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_proba = np.mean(np.log(predictions + epsilon), axis=0)
        ensemble_proba = np.exp(log_proba)
        
    elif method == 'rank_mean':
        # Rank-transform predictions, then average ranks, then convert back
        n_samples = predictions.shape[1]
        rank_predictions = np.zeros_like(predictions)
        for i in range(n_samples):
            ranks = np.array([np.searchsorted(np.sort(pred), pred[i]) for pred in predictions])
            rank_predictions[:, i] = ranks
        # Normalize ranks to [0, 1]
        rank_predictions = rank_predictions / (n_samples - 1) if n_samples > 1 else rank_predictions
        ensemble_proba = np.mean(rank_predictions, axis=0)
        
    elif method == 'vote':
        # Majority voting with tie-breaking
        binary_preds = (predictions > 0.5).astype(int)
        vote_sum = np.sum(binary_preds, axis=0)
        majority_threshold = n_models / 2.0
        
        # Majority wins
        ensemble_binary = (vote_sum > majority_threshold).astype(int)
        
        # Handle ties (exactly half vote positive)
        ties = (vote_sum == majority_threshold)
        n_ties = ties.sum()
        
        if n_ties > 0:
            print(f"   Ties detected: {n_ties} samples")
            if tie_breaker == 'conservative':
                # Predict negative on ties
                ensemble_binary[ties] = 0
                print(f"   Tie-breaking: Conservative (predict negative)")
            elif tie_breaker == 'aggressive':
                # Predict positive on ties
                ensemble_binary[ties] = 1
                print(f"   Tie-breaking: Aggressive (predict positive)")
            elif tie_breaker == 'prob_mean':
                # Use mean probability to break ties
                tie_proba = np.mean(predictions[:, ties], axis=0)
                ensemble_binary[ties] = (tie_proba >= 0.5).astype(int)
                print(f"   Tie-breaking: Probability mean")
            else:
                raise ValueError(f"Unknown tie_breaker: {tie_breaker}")
        
        # Convert binary to probabilities (use vote fraction as probability)
        ensemble_proba = vote_sum / n_models
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble_proba


def break_ties_binary(
    predictions: np.ndarray,
    threshold: float = 0.5,
    tie_breaker: str = 'conservative'
) -> np.ndarray:
    """
    Break ties when predictions are exactly at threshold.
    
    Args:
        predictions: Array of probabilities
        threshold: Classification threshold
        tie_breaker: How to handle exact ties:
            - 'conservative': Predict negative
            - 'aggressive': Predict positive
            - 'round_up': Round up (predict positive)
            - 'round_down': Round down (predict negative)
    
    Returns:
        Binary predictions
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    # Find exact ties
    ties = (predictions == threshold)
    n_ties = ties.sum()
    
    if n_ties > 0:
        print(f"   Threshold ties detected: {n_ties} samples")
        if tie_breaker == 'conservative':
            binary_preds[ties] = 0
        elif tie_breaker == 'aggressive':
            binary_preds[ties] = 1
        elif tie_breaker == 'round_up':
            binary_preds[ties] = 1
        elif tie_breaker == 'round_down':
            binary_preds[ties] = 0
        else:
            raise ValueError(f"Unknown tie_breaker: {tie_breaker}")
        print(f"   Tie-breaking: {tie_breaker}")
    
    return binary_preds


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scalable ensemble prediction from saved models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic ensemble with all models
  python3 ensemble_predict.py

  # Weighted ensemble with specific models
  python3 ensemble_predict.py --models model_xgboost model_lightgbm model_catboost

  # Geometric mean with conservative tie-breaking
  python3 ensemble_predict.py --method geometric_mean --tie-breaker conservative

  # Majority voting with probability-based tie-breaking
  python3 ensemble_predict.py --method vote --tie-breaker prob_mean
        """
    )
    
    parser.add_argument('--test-data', type=str,
                       default='data/model_ready/test_model_ready.parquet',
                       help='Path to test data parquet file')
    parser.add_argument('--models-dir', type=str,
                       default='models/saved_models',
                       help='Directory containing saved model .pkl files')
    parser.add_argument('--output', type=str,
                       default='data/submission_files/submission_ensemble.csv',
                       help='Output submission file path')
    parser.add_argument('--models', type=str, nargs='*',
                       default=None,
                       help='Specific model files to use (optional, otherwise uses all)')
    parser.add_argument('--method', type=str,
                       choices=['mean', 'weighted_mean', 'geometric_mean', 'rank_mean', 'vote'],
                       default='weighted_mean',
                       help='Ensemble method')
    parser.add_argument('--threshold', type=float,
                       default=None,
                       help='Threshold for binary classification (auto if None)')
    parser.add_argument('--tie-breaker', type=str,
                       choices=['conservative', 'aggressive', 'prob_mean', 'round_up', 'round_down'],
                       default='conservative',
                       help='Tie-breaking strategy')
    parser.add_argument('--device', type=str,
                       default='cpu',
                       help='Device for PyTorch models (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Setup paths (script is in src/scripts/, so go up two levels to project root)
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / args.models_dir
    test_path = project_root / args.test_data
    output_path = project_root / args.output
    
    print("=" * 80)
    print("SCALABLE ENSEMBLE PREDICTION")
    print("=" * 80)
    print(f"Models directory: {models_dir}")
    print(f"Test data: {test_path}")
    print(f"Output: {output_path}")
    print(f"Ensemble method: {args.method}")
    print(f"Tie-breaker: {args.tie_breaker}")
    print()
    
    # Check paths
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Find model files
    if args.models:
        model_files = [models_dir / f"{m}_all_features_best.pkl" if not m.endswith('.pkl') else models_dir / m
                      for m in args.models]
    else:
        model_files = sorted(models_dir.glob('model_*_all_features_best.pkl'))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    print(f"Found {len(model_files)} model file(s):")
    for mf in model_files:
        print(f"  - {mf.name}")
    print()
    
    # Load models
    models = []
    for model_file in model_files:
        if not model_file.exists():
            print(f"⚠️  Model file not found: {model_file}")
            continue
        
        try:
            model_data = load_saved_model(model_file)
            model_data['model_name'] = model_file.stem.replace('_all_features_best', '')
            models.append(model_data)
            
            # Determine model type
            is_pytorch = 'model_state_dict' in model_data
            model_type = 'PyTorch MLP' if is_pytorch else 'sklearn'
            print(f"   Model type: {model_type}")
            
            # Show performance metrics
            best_f1 = model_data.get('best_f1')
            best_cv_score = model_data.get('best_cv_score')
            if best_f1:
                print(f"   Best F1: {best_f1:.4f}")
            elif best_cv_score:
                print(f"   CV F1: {best_cv_score:.4f}")
            print()
            
        except Exception as e:
            print(f"⚠️  Error loading {model_file.name}: {e}")
            continue
    
    if not models:
        raise ValueError("No valid models loaded!")
    
    print(f"✅ Loaded {len(models)} valid model(s)")
    print()
    
    # Determine chunk size based on available memory
    # Process in chunks to avoid OOM
    chunk_size = 20000  # Process 20k samples at a time
    
    # Get total number of rows efficiently
    test_df_sample = pl.read_parquet(test_path, n_rows=1)
    has_id_col = 'id' in test_df_sample.columns
    
    # Use streaming to get row count efficiently
    n_total = pl.scan_parquet(test_path).select(pl.len()).collect().item()
    
    print(f"📊 Test data: {n_total:,} samples")
    print(f"   Processing in chunks of {chunk_size:,} samples")
    print()
    
    # Extract work IDs efficiently (read only id column)
    if has_id_col:
        print("📋 Extracting work IDs...")
        work_ids_df = pl.scan_parquet(test_path).select(['id']).collect()
        work_ids = np.array([extract_work_id(str(id_val)) for id_val in work_ids_df['id'].to_numpy()])
        del work_ids_df
    else:
        work_ids = np.array([f'W{i}' for i in range(n_total)])
        print("⚠️  No 'id' column found. Using sequential work IDs.")
    
    # Get predictions from each model (chunked processing)
    print("=" * 80)
    print("GETTING PREDICTIONS FROM EACH MODEL (CHUNKED)")
    print("=" * 80)
    
    # Store predictions for each model
    all_model_predictions = {i: [] for i in range(len(models))}
    
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    
    # Use streaming for efficient chunked reading
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_total)
        chunk_size_actual = end_idx - start_idx
        
        print(f"\n📦 Processing chunk {chunk_idx + 1}/{n_chunks} (rows {start_idx:,}-{end_idx:,})")
        
        # Load chunk using streaming with slice
        test_df_chunk = pl.scan_parquet(test_path).slice(start_idx, chunk_size_actual).collect()
        
        # Process each model
        for i, model_data in enumerate(models):
            model_name = model_data.get('model_name', f'Model_{i}')
            
            try:
                # Determine preprocessing
                is_pytorch = 'model_state_dict' in model_data
                
                # Prepare features for this chunk
                if is_pytorch:
                    X_chunk = prepare_features_pytorch(test_df_chunk, model_data)
                else:
                    X_chunk = prepare_features_sklearn(test_df_chunk, model_data)
                
                # Get predictions for this chunk
                if is_pytorch:
                    if not TORCH_AVAILABLE:
                        if chunk_idx == 0:
                            print(f"   ⚠️ Skipping {model_name}: PyTorch not available")
                        continue
                    y_proba_chunk = predict_pytorch_model(model_data, X_chunk, device=args.device)
                else:
                    y_proba_chunk = predict_sklearn_model(model_data, X_chunk)
                
                all_model_predictions[i].append(y_proba_chunk)
                
                if chunk_idx == 0:
                    print(f"   [{i+1}/{len(models)}] {model_name}: chunk shape {y_proba_chunk.shape}")
                
                # Cleanup
                del X_chunk, y_proba_chunk
                
            except Exception as e:
                print(f"   ❌ Error in {model_name}, chunk {chunk_idx + 1}: {e}")
                if chunk_idx == 0:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Cleanup chunk
        del test_df_chunk
        
        # Memory cleanup
        import gc
        gc.collect()
    
    # Concatenate predictions for each model
    print("\n📊 Concatenating predictions...")
    model_predictions = []
    for i, model_data in enumerate(models):
        if i not in all_model_predictions or not all_model_predictions[i]:
            continue
        
        y_proba_full = np.concatenate(all_model_predictions[i])
        model_predictions.append((y_proba_full, model_data))
        
        model_name = model_data.get('model_name', f'Model_{i}')
        print(f"   ✅ {model_name}: {y_proba_full.shape}, Mean: {y_proba_full.mean():.4f}")
    
    if not model_predictions:
        raise ValueError("No successful predictions!")
    
    print(f"\n✅ Got predictions from {len(model_predictions)} model(s)")
    print()
    
    # Make ensemble predictions
    print("=" * 80)
    print("ENSEMBLE PREDICTIONS")
    print("=" * 80)
    
    ensemble_proba = ensemble_predict(
        model_predictions,
        method=args.method,
        tie_breaker=args.tie_breaker
    )
    
    print(f"   Ensemble probability shape: {ensemble_proba.shape}")
    print(f"   Mean probability: {ensemble_proba.mean():.4f}")
    print(f"   Std probability: {ensemble_proba.std():.4f}")
    print(f"   Min/Max: {ensemble_proba.min():.4f} / {ensemble_proba.max():.4f}")
    print()
    
    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"   Using provided threshold: {threshold:.4f}")
    else:
        # Use average of model thresholds, or default
        thresholds = [md.get('best_threshold') for _, md in model_predictions
                     if md.get('best_threshold') is not None]
        if thresholds:
            threshold = np.mean(thresholds)
            print(f"   Using average model threshold: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"   Using default threshold: {threshold:.4f}")
    
    # Apply threshold with tie-breaking
    binary_preds = break_ties_binary(
        ensemble_proba,
        threshold=threshold,
        tie_breaker=args.tie_breaker
    )
    
    print(f"   Binary predictions: Positive={binary_preds.sum()}, Negative={(binary_preds==0).sum()}")
    print()
    
    # Create submission
    print("=" * 80)
    print("CREATING SUBMISSION FILE")
    print("=" * 80)
    
    submission_df = pl.DataFrame({
        'work_id': work_ids,
        'label': binary_preds
    })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.write_csv(output_path)
    
    print(f"✅ Submission saved to: {output_path}")
    print(f"   Total predictions: {len(binary_preds)}")
    print(f"   Positive: {binary_preds.sum()} ({100*binary_preds.sum()/len(binary_preds):.2f}%)")
    print(f"   Negative: {(binary_preds==0).sum()} ({100*(binary_preds==0).sum()/len(binary_preds):.2f}%)")
    print()
    
    print("=" * 80)
    print("✅ ENSEMBLE PREDICTION COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
