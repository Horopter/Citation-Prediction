"""
Shared utilities for model training with CV, hyperparameter tuning, threshold tuning, and submission generation.
"""

import os
import gc
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import json

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

# Import PyTorch PCA utilities
try:
    from .pca_utils import IncrementalTorchPCA
except ImportError:
    # Fallback if relative import fails
    import sys
    from pathlib import Path
    utils_path = Path(__file__).parent
    sys.path.insert(0, str(utils_path))
    from pca_utils import IncrementalTorchPCA

# Memory management utilities (shared across notebooks)
def cleanup_memory():
    """Aggressive memory cleanup for both CPU and GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()  # Second pass for thorough cleanup


def memory_usage():
    """Display current memory usage statistics."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"💾 Memory: {mem_info.rss / 1024**3:.2f} GB (RAM)", end="")
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f" | {gpu_mem:.2f}/{gpu_reserved:.2f} GB (GPU used/reserved)")
        else:
            print()
    except ImportError:
        print("💾 Memory tracking requires psutil: pip install psutil")


def get_memory_usage():
    """Get current memory usage as a dict (for programmatic use)."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        result = {'ram_gb': mem_info.rss / 1024**3}
        
        if torch.cuda.is_available():
            result['gpu_used_gb'] = torch.cuda.memory_allocated() / 1024**3
            result['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            result['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            result['gpu_used_gb'] = 0
            result['gpu_reserved_gb'] = 0
            result['gpu_total_gb'] = 0
        
        return result
    except ImportError:
        return {'ram_gb': 0, 'gpu_used_gb': 0, 'gpu_reserved_gb': 0, 'gpu_total_gb': 0}


def check_memory_safe(ram_threshold_gb=0.9, gpu_threshold=0.85):
    """
    Check if memory usage is safe (below thresholds).
    
    Args:
        ram_threshold_gb: RAM threshold in GB (default: 90% of available)
        gpu_threshold: GPU threshold as fraction (default: 85%)
    
    Returns:
        (is_safe, memory_info_dict)
    """
    try:
        import psutil
        mem_info = get_memory_usage()
        
        # Get total RAM
        total_ram_gb = psutil.virtual_memory().total / 1024**3
        ram_usage_ratio = mem_info['ram_gb'] / total_ram_gb if total_ram_gb > 0 else 0
        
        # Check RAM
        ram_safe = ram_usage_ratio < ram_threshold_gb
        
        # Check GPU
        gpu_safe = True
        if torch.cuda.is_available() and mem_info['gpu_total_gb'] > 0:
            gpu_usage_ratio = mem_info['gpu_reserved_gb'] / mem_info['gpu_total_gb']
            gpu_safe = gpu_usage_ratio < gpu_threshold
        
        is_safe = ram_safe and gpu_safe
        
        if not is_safe:
            gpu_str = f"{gpu_usage_ratio:.1%}" if torch.cuda.is_available() and mem_info['gpu_total_gb'] > 0 else "N/A"
            print(f"⚠️ Memory warning: RAM={ram_usage_ratio:.1%}, GPU={gpu_str}")
        
        return is_safe, mem_info
    except Exception:
        # If we can't check, assume safe (better than crashing)
        return True, get_memory_usage()


def adaptive_batch_size(current_batch_size, min_batch_size=32):
    """
    Suggest a reduced batch size if memory is getting high.
    
    Args:
        current_batch_size: Current batch size
        min_batch_size: Minimum allowed batch size
    
    Returns:
        Suggested batch size
    """
    is_safe, mem_info = check_memory_safe()
    
    if not is_safe:
        # Reduce batch size by half, but not below minimum
        suggested = max(min_batch_size, current_batch_size // 2)
        if suggested < current_batch_size:
            print(f"⚠️ Memory pressure detected. Suggesting batch size reduction: {current_batch_size} -> {suggested}")
        return suggested
    
    return current_batch_size


def safe_training_step(model, xb, yb, criterion, optimizer, device):
    """
    Safe training step with memory protection.
    
    Returns:
        loss value or None if OOM occurred
    """
    try:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        
        # Clean up immediately
        del xb, yb, logits, loss
        
        return loss_val
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"⚠️ OOM detected in training step. Cleaning up...")
            cleanup_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        raise


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, 
                          thresholds: np.ndarray = None) -> Tuple[float, float]:
    """
    Find optimal threshold for F1 score.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        thresholds: Array of thresholds to test (default: 0.1 to 0.9, step 0.05)
    
    Returns:
        Tuple of (best_threshold, best_f1_score)
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thr in thresholds:
        y_pred = (y_probs >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
    
    return best_threshold, best_f1


def train_fold(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               criterion: nn.Module, optimizer: torch.optim.Optimizer,
               device: torch.device, epochs: int, early_stopping_patience: int = 5,
               verbose: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Train a model for one fold with early stopping.
    
    Returns:
        Tuple of (best_model_state_dict, history_dict, best_threshold)
    """
    best_val_f1 = 0.0
    best_state_dict = None
    best_threshold = 0.5  # Default threshold
    patience_counter = 0
    history = {'train_loss': [], 'val_f1': [], 'val_roc_auc': [], 'val_pr_auc': [], 'threshold': []}
    
    for epoch in range(1, epochs + 1):
        # Check memory before epoch
        is_safe, _ = check_memory_safe()
        if not is_safe:
            cleanup_memory()
        
        # Training
        model.train()
        running_loss = 0.0
        batch_count = 0
        successful_batches = 0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            # Periodic memory check every 10 batches
            if batch_idx % 10 == 0:
                is_safe, _ = check_memory_safe()
                if not is_safe:
                    cleanup_memory()
            
            # Safe training step with OOM protection
            loss_val = safe_training_step(model, xb, yb, criterion, optimizer, device)
            
            if loss_val is not None:
                running_loss += loss_val * xb.size(0)
                successful_batches += 1
            else:
                # OOM occurred - skip this batch and cleanup
                print(f"  ⚠️ Skipping batch {batch_idx} due to OOM")
                cleanup_memory()
                continue
            
            batch_count += 1
            
            # Periodic cleanup every 50 batches
            if batch_idx % 50 == 0:
                cleanup_memory()
        
        if successful_batches == 0:
            print(f"  ⚠️ Warning: No successful batches in epoch {epoch}")
            cleanup_memory()
            continue
        
        avg_train_loss = running_loss / len(train_loader.dataset) if successful_batches > 0 else float('inf')
        
        # Validation with memory protection
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(val_loader):
                # Periodic memory check
                if batch_idx % 20 == 0:
                    is_safe, _ = check_memory_safe()
                    if not is_safe:
                        cleanup_memory()
                
                try:
                    xb = xb.to(device, non_blocking=True)
                    yb_np = yb.numpy()
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy().ravel()
                    all_preds.append(probs)
                    all_targets.append(yb_np)
                    del xb, logits, probs, yb_np
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"  ⚠️ OOM in validation batch {batch_idx}, cleaning up...")
                        cleanup_memory()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Skip this batch
                        continue
                    raise
                
                # Periodic cleanup
                if batch_idx % 50 == 0:
                    cleanup_memory()
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        # Metrics
        val_threshold, val_f1 = find_optimal_threshold(all_targets, all_preds)
        val_roc_auc = roc_auc_score(all_targets, all_preds)
        val_pr_auc = average_precision_score(all_targets, all_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)
        history['val_pr_auc'].append(val_pr_auc)
        history['threshold'].append(val_threshold)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = val_threshold
            best_state_dict = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break
        
        cleanup_memory()
    
    # Ensure we always return a valid state_dict
    if best_state_dict is None:
        # Fallback: return current model state if no improvement was found
        best_state_dict = model.state_dict()
        # Use last threshold if no improvement
        if history['threshold']:
            best_threshold = history['threshold'][-1]
    
    return best_state_dict, history, best_threshold


def stratified_kfold_splits(y: np.ndarray, n_splits: int = 5, shuffle: bool = True, 
                            random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    PyTorch-friendly stratified k-fold splits.
    
    Args:
        y: Labels array
        n_splits: Number of folds
        shuffle: Whether to shuffle
        random_state: Random seed
    
    Returns:
        List of (train_idx, val_idx) tuples
    """
    np.random.seed(random_state)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    
    # Get indices for each class
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}
    
    # Shuffle each class's indices if requested
    if shuffle:
        for cls in class_indices:
            np.random.shuffle(class_indices[cls])
    
    splits = []
    for fold in range(n_splits):
        train_idx_list = []
        val_idx_list = []
        
        for cls in unique_classes:
            indices = class_indices[cls]
            n_samples = len(indices)
            fold_size = n_samples // n_splits
            start = fold * fold_size
            end = start + fold_size if fold < n_splits - 1 else n_samples
            
            val_idx_list.extend(indices[start:end])
            train_idx_list.extend(np.concatenate([indices[:start], indices[end:]]))
        
        train_idx = np.array(train_idx_list, dtype=np.int64)
        val_idx = np.array(val_idx_list, dtype=np.int64)
        
        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(val_idx)
        
        splits.append((train_idx, val_idx))
    
    return splits


def cross_validate(model_class: type, model_kwargs: Dict, X_train: np.ndarray, y_train: np.ndarray,
                  make_dataloaders_func, criterion: nn.Module, optimizer_class: type,
                  optimizer_kwargs: Dict, device: torch.device, n_splits: int = 5,
                  epochs: int = 20, early_stopping_patience: int = 5,
                  verbose: bool = True, checkpoint_dir: Path = None) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation using PyTorch-friendly splits.
    
    Args:
        checkpoint_dir: Optional directory to save CV checkpoints (for recovery)
    
    Returns:
        Dictionary with CV results and best model
    """
    # Initial memory check
    is_safe, mem_info = check_memory_safe()
    if not is_safe:
        print("⚠️ High memory usage detected at CV start. Cleaning up...")
        cleanup_memory()
    
    splits = stratified_kfold_splits(y_train, n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_fold_f1 = 0.0
    best_fold_model_state = None
    best_fold_idx = -1
    
    # Create checkpoint directory if provided
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        # Kernel heartbeat - keep kernel alive
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}/{n_splits} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            memory_usage()
        
        # Memory check before fold
        is_safe, _ = check_memory_safe()
        if not is_safe:
            print("⚠️ Memory pressure before fold. Cleaning up...")
            cleanup_memory()
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Cleanup after indexing
        cleanup_memory()
        
        # Create dataloaders
        train_loader, val_loader = make_dataloaders_func(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val
        )
        
        # Memory check after dataloader creation
        is_safe, _ = check_memory_safe()
        if not is_safe:
            print("⚠️ Memory pressure after dataloader creation. Cleaning up...")
            cleanup_memory()
        
        # Create model
        model = model_class(**model_kwargs).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Train fold
        start_time = time.time()
        best_state_dict, history, best_threshold = train_fold(
            model, train_loader, val_loader, criterion, optimizer,
            device, epochs, early_stopping_patience, verbose
        )
        fold_time = time.time() - start_time
        
        # Extract best metrics from history (already computed in train_fold)
        best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
        best_val_roc_auc = max(history['val_roc_auc']) if history['val_roc_auc'] else 0.0
        best_val_pr_auc = max(history['val_pr_auc']) if history['val_pr_auc'] else 0.0
        
        fold_results.append({
            'fold': fold_idx,
            'val_f1': best_val_f1,
            'val_roc_auc': best_val_roc_auc,
            'val_pr_auc': best_val_pr_auc,
            'threshold': best_threshold,
            'history': history,
            'time': fold_time
        })
        
        if best_state_dict is None:
            if verbose:
                print(f"  ⚠️ WARNING: CV fold {fold_idx} returned None state_dict")
            best_state_dict = {}  # Fallback to empty dict
        
        if best_val_f1 > best_fold_f1:
            best_fold_f1 = best_val_f1
            best_fold_model_state = best_state_dict
            best_fold_idx = fold_idx
        
        # Save checkpoint after each fold
        if checkpoint_dir:
            checkpoint_path = checkpoint_dir / f"cv_fold_{fold_idx}_checkpoint.pt"
            try:
                torch.save({
                    'fold_idx': fold_idx,
                    'best_state_dict': best_state_dict,
                    'best_fold_f1': best_fold_f1,
                    'best_fold_idx': best_fold_idx,
                    'fold_results': fold_results,
                    'timestamp': time.time()
                }, checkpoint_path)
                if verbose:
                    print(f"  💾 Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"  ⚠️ Failed to save checkpoint: {e}")
        
        if verbose:
            print(f"  Fold {fold_idx} - Val F1: {best_val_f1:.4f}, ROC-AUC: {best_val_roc_auc:.4f}, PR-AUC: {best_val_pr_auc:.4f}, Time: {fold_time:.1f}s")
        
        # Cleanup after fold
        del model, optimizer, train_loader, val_loader
        cleanup_memory()
        
        # Periodic cleanup and memory check
        is_safe, _ = check_memory_safe()
        if not is_safe:
            print("  ⚠️ Memory pressure after fold. Performing aggressive cleanup...")
            cleanup_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Aggregate results
    cv_results = {
        'fold_results': fold_results,
        'mean_f1': np.mean([r['val_f1'] for r in fold_results]),
        'std_f1': np.std([r['val_f1'] for r in fold_results]),
        'mean_roc_auc': np.mean([r['val_roc_auc'] for r in fold_results]),
        'mean_pr_auc': np.mean([r['val_pr_auc'] for r in fold_results]),
        'mean_threshold': np.mean([r['threshold'] for r in fold_results]),
        'best_fold': best_fold_idx,
        'best_fold_f1': best_fold_f1,
        'best_model_state': best_fold_model_state
    }
    
    return cv_results


def generate_submission(model: nn.Module, test_loader: DataLoader, test_ids: np.ndarray,
                       device: torch.device, threshold: float,
                       output_path: Path) -> None:
    """
    Generate submission.csv file.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        test_ids: Array of test IDs
        device: torch device
        threshold: Classification threshold
        output_path: Path to save submission.csv
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch_idx, (xb, _) in enumerate(test_loader):
            # Periodic memory check
            if batch_idx % 20 == 0:
                is_safe, _ = check_memory_safe()
                if not is_safe:
                    cleanup_memory()
            
            try:
                xb = xb.to(device, non_blocking=True)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                all_preds.append(probs)
                del xb, logits, probs
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"⚠️ OOM in test batch {batch_idx}, cleaning up...")
                    cleanup_memory()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Skip this batch (use zeros as fallback)
                    all_preds.append(np.zeros(xb.size(0)))
                    continue
                raise
            
            # Periodic cleanup
            if batch_idx % 50 == 0:
                cleanup_memory()
    
    all_preds = np.concatenate(all_preds)
    predictions = (all_preds >= threshold).astype(int)
    
    # Extract work_id from URLs (format: https://openalex.org/W123456789 -> W123456789)
    work_ids = []
    for test_id in test_ids:
        if isinstance(test_id, str) and 'openalex.org/W' in test_id:
            # Extract work ID from URL
            match = re.search(r'W\d+', test_id)
            if match:
                work_ids.append(match.group())
            else:
                work_ids.append(test_id)  # Fallback to original if no match
        else:
            # Already a work ID or other format
            work_ids.append(str(test_id))
    
    # Create submission DataFrame using Polars
    submission_df = pl.DataFrame({
        'work_id': work_ids,
        'label': predictions
    })
    
    submission_df.write_csv(output_path)
    print(f"\n✅ Submission saved to: {output_path}")
    print(f"   Predictions: {predictions.sum()} positive, {len(predictions) - predictions.sum()} negative")


def save_model_weights(model: nn.Module, save_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Save model weights with metadata.
    
    Args:
        model: Trained model
        save_path: Path to save model
        metadata: Dictionary of metadata to save
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        **metadata
    }, save_path)
    
    print(f"💾 Model weights saved to: {save_path}")

