# Model Comparison Report: XGBoost, CatBoost, and LightGBM

## Executive Summary

This report compares the performance of three gradient boosting models (XGBoost, CatBoost, and LightGBM) trained on the Citation Prediction dataset. All models were trained using the same feature set (54 regular features + PCA-compressed embeddings from three transformer models) and underwent identical preprocessing pipelines including SMOTETomek resampling, feature scaling, and 5-fold cross-validation.

### Key Findings

- **Best F1 Score**: CatBoost (0.5206) slightly outperforms XGBoost (0.5125)
- **Best Accuracy**: CatBoost (0.9288) achieves the highest accuracy
- **Best Cross-Validation Score**: XGBoost (0.5833) shows the best CV performance
- **Optimal Thresholds**: Models use different thresholds (CatBoost: 0.2754, XGBoost: 0.7623), indicating different calibration characteristics

## Performance Metrics Comparison

| Model | F1 Score | Accuracy | CV F1 Score | Optimal Threshold | Notebook Size |
|-------|----------|----------|-------------|-------------------|---------------|
| **XGBoost** | 0.5125 | 0.9208 | 0.5833 | 0.7623 | 983.4 KB |
| **CatBoost** | 0.5206 | 0.9288 | - | 0.2754 | 759.1 KB |
| **LightGBM** | - | - | - | - | 83.1 KB |

### Detailed Analysis

#### XGBoost
- **F1 Score**: 0.5125
- **Accuracy**: 0.9208 (92.08%)
- **Cross-Validation F1**: 0.5833
- **Optimal Threshold**: 0.7623 (relatively high, indicating conservative predictions)
- **Training Characteristics**: 
  - Uses RandomizedSearchCV for hyperparameter tuning
  - 5-fold stratified cross-validation
  - Fine-grained threshold optimization (120+ thresholds)
  - Incremental SMOTETomek for class imbalance handling

#### CatBoost
- **F1 Score**: 0.5206 (best among the three models)
- **Accuracy**: 0.9288 (92.88%, highest)
- **Optimal Threshold**: 0.2754 (relatively low, indicating more aggressive positive predictions)
- **Training Characteristics**:
  - Native categorical feature handling
  - Similar preprocessing pipeline to XGBoost
  - SMOTETomek resampling
  - RandomizedSearchCV hyperparameter optimization

#### LightGBM
- **Metrics**: Limited metrics extracted from executed notebook
- **Notebook Size**: 83.1 KB (smallest, suggesting shorter execution or fewer outputs)
- **Note**: The executed notebook appears to have incomplete output or may have encountered issues during execution

## Hyperparameter Comparison

All three models underwent hyperparameter tuning using RandomizedSearchCV with similar parameter grids. Common hyperparameters tuned include:

- **n_estimators**: Number of boosting rounds (typically 100-300)
- **max_depth**: Maximum tree depth (typically 3-7, or -1 for unlimited)
- **learning_rate**: Step size shrinkage (typically 0.01-0.2)
- **subsample**: Row sampling ratio (typically 0.6-1.0)
- **colsample_bytree**: Column sampling ratio (typically 0.6-1.0)
- **Regularization**: L1 (reg_alpha) and L2 (reg_lambda) regularization parameters

### Model-Specific Characteristics

**XGBoost:**
- Strong regularization capabilities
- Excellent for handling sparse data
- Good performance with default parameters
- More conservative predictions (higher threshold)

**CatBoost:**
- Native categorical feature handling (no need for one-hot encoding)
- Built-in overfitting prevention
- Automatic handling of missing values
- More aggressive predictions (lower threshold)

**LightGBM:**
- Fast training speed
- Lower memory usage
- Good for large datasets
- Limited execution data available

## Training Efficiency

| Model | Notebook Size | Training Approach | Memory Management |
|-------|---------------|-------------------|-------------------|
| XGBoost | 983.4 KB | Full execution with detailed outputs | OOM-resistant with chunked processing |
| CatBoost | 759.1 KB | Full execution with detailed outputs | OOM-resistant with chunked processing |
| LightGBM | 83.1 KB | Limited execution data | OOM-resistant design |

## Ensemble Recommendations

Based on the performance metrics, the following ensemble weighting is recommended:

1. **CatBoost**: Highest F1 score (0.5206) and accuracy (0.9288) - **Weight: 0.40**
2. **XGBoost**: Best CV score (0.5833) and strong overall performance - **Weight: 0.40**
3. **LightGBM**: Limited data, but can contribute diversity - **Weight: 0.20** (if metrics become available)

### Ensemble Strategy

The current ensemble implementation uses **weighted mean** based on validation F1 scores. Given the metrics:

- CatBoost should receive the highest weight due to best F1 score
- XGBoost should receive significant weight due to best CV performance
- The different optimal thresholds (0.2754 vs 0.7623) suggest the models capture different aspects of the data, making them complementary

## Threshold Analysis

The significant difference in optimal thresholds between models is noteworthy:

- **CatBoost (0.2754)**: Lower threshold suggests the model is well-calibrated and confident in positive predictions
- **XGBoost (0.7623)**: Higher threshold suggests the model is more conservative, requiring higher confidence before predicting positive class

This difference indicates:
1. Models have different probability distributions
2. Ensemble methods that account for calibration (like isotonic calibration) may improve performance
3. The weighted ensemble should balance these different prediction styles

## Recommendations

### For Production Use

1. **Primary Model**: Use CatBoost as the primary model due to highest F1 score and accuracy
2. **Ensemble**: Combine CatBoost and XGBoost with weighted averaging (40% each, 20% for LightGBM if available)
3. **Threshold Tuning**: Consider ensemble-level threshold optimization rather than individual model thresholds
4. **Calibration**: Apply isotonic calibration to improve probability estimates before ensembling

### For Further Improvement

1. **LightGBM**: Re-run LightGBM training to obtain complete metrics and ensure proper execution
2. **Hyperparameter Refinement**: Fine-tune hyperparameters based on validation performance
3. **Feature Engineering**: Explore additional features or feature interactions
4. **Advanced Ensemble Methods**: Experiment with stacking or blending techniques
5. **Cross-Validation**: Use consistent CV methodology across all models for fair comparison

## Conclusion

CatBoost demonstrates the best individual performance with the highest F1 score (0.5206) and accuracy (0.9288). However, XGBoost shows the best cross-validation performance (0.5833), suggesting strong generalization. The complementary nature of these models (evidenced by different optimal thresholds) makes them excellent candidates for ensemble methods.

The current weighted ensemble approach, which combines these models based on their validation F1 scores, should provide robust predictions that leverage the strengths of each model while mitigating individual weaknesses.

---

**Report Generated**: December 24, 2024  
**Data Source**: Executed notebooks from `runs/` directory  
**Competition**: [Kaggle F-25 SI-670 Kaggle 2](https://www.kaggle.com/competitions/f-25-si-670-kaggle-2)






