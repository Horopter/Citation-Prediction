# Feature Set Comparison: XGBoost Model vs Multi-Classifier Model

## Overview

This document compares the feature sets used in:
- **XGBoost Model**: `model_xgboost_all_features.ipynb`
- **Multi-Classifier Model**: `model_multi_classifier_leakage_test.ipynb`

---

## Feature Summary

### XGBoost Model Features

**Total Features: ~147 features**

1. **Regular Features (51 features)** - From `data_exploration_organized.ipynb`:
   - Basic Metadata (3): `publication_year`, `type`, `language`
   - Text Features (5): `title_length`, `abstract_length`, `has_abstract`, `title_word_count`, `abstract_word_count`
   - Author Features (10): `num_authors`, `num_institutions`, `first_author_h_index`, `first_author_citations`, `first_author_papers`, `max_author_h_index`, `avg_author_h_index`, `total_author_citations`, `avg_author_citations`
   - Venue Features (5): `venue_impact_factor`, `venue_h_index`, `venue_citations`, `is_oa_venue`, `is_in_doaj`
   - Concept Features (4): `num_concepts`, `max_concept_score`, `avg_concept_score`, `num_high_concepts`
   - Reference Features (2): `num_references`, `num_related_works`
   - Identifier Features (3): `has_doi`, `has_pmid`, `has_pmcid`
   - Grant Features (2): `num_grants`, `has_grants`
   - Location/Topic/Keyword/Mesh/SDG Features (8): `num_locations`, `num_topics`, `max_topic_score`, `avg_topic_score`, `num_keywords`, `num_mesh_terms`, `num_sdgs`, `has_sdgs`
   - NLP Features (8): `nlp_char_count`, `nlp_word_count`, `nlp_avg_word_length`, `nlp_sentence_count`, `nlp_avg_sentence_length`, `nlp_capital_ratio`, `nlp_number_ratio`, `nlp_punctuation_ratio`
   - Open Access Features (1): `is_oa` (plus `oa_status_{status}` one-hot encoded)

2. **Embedding Features (96 features)** - PCA-compressed:
   - Sentence Transformers: 32 PCA components (`sent_transformer_pca_0` to `sent_transformer_pca_31`)
   - SciBERT: 32 PCA components (`scibert_pca_0` to `scibert_pca_31`)
   - SPECTER2: 32 PCA components (`specter2_pca_0` to `specter2_pca_31`)

**Preprocessing:**
- ✅ PCA compression: 32 components per embedding family
- ✅ StandardScaler: Applied to ALL features
- ✅ SMOTETomek/SMOTEENN: Incremental resampling (10k chunks)
- ✅ No temporal features
- ✅ No additional feature engineering beyond base features

---

### Multi-Classifier Model Features

**Total Features: ~150 features** (after duplicate elimination)

1. **Regular Features (51 features)** - Same as XGBoost:
   - Identical to XGBoost's regular features (from `data_exploration_organized.ipynb`)

2. **Embedding Features (96 features)** - Same as XGBoost:
   - Sentence Transformers: 32 PCA components
   - SciBERT: 32 PCA components
   - SPECTER2: 32 PCA components

3. **Temporal Features (3 features)** - **ADDED**:
   - `num_years_after_publication` = 2025 - publication_year
   - `days_since_updated` = (datetime.now() - updated_date).total_days()
   - `days_since_publication` = (datetime.now() - publication_date).total_days()

**Preprocessing:**
- ✅ PCA compression: 32 components per embedding family (same as XGBoost)
- ✅ Selective StandardScaler: Applied ONLY to numeric columns (preserves binary/one-hot features)
- ✅ RandomUnderSampler: For class imbalance (different from XGBoost's SMOTE)
- ✅ Duplicate elimination: Removes absolute duplicate features
- ✅ Temporal feature engineering: Adds 3 date-derived features

---

## Detailed Comparison

### ✅ Features Present in Both

| Feature Category | XGBoost | Multi-Classifier | Status |
|-----------------|---------|------------------|--------|
| Regular features (51) | ✅ | ✅ | ✅ **MATCH** |
| Embedding features (96) | ✅ | ✅ | ✅ **MATCH** |
| PCA compression | ✅ (32 per family) | ✅ (32 per family) | ✅ **MATCH** |

**Total Common Features: 147 features**

---

### ➕ Features Only in Multi-Classifier Model

| Feature | Description | Impact |
|---------|-------------|--------|
| `num_years_after_publication` | Years since publication (2025 - publication_year) | **HIGH** - Temporal leakage feature |
| `days_since_updated` | Days since last update | **HIGH** - Temporal leakage feature |
| `days_since_publication` | Days since publication | **HIGH** - Temporal leakage feature |

**Additional Features: +3 features**

---

### ⚠️ Preprocessing Differences

| Aspect | XGBoost | Multi-Classifier | Impact |
|--------|---------|------------------|--------|
| **Feature Scaling** | StandardScaler on ALL features | Selective scaling (numeric only) | **MODERATE** - Preserves binary features better |
| **Class Imbalance** | SMOTETomek/SMOTEENN (incremental, 10k chunks) | RandomUnderSampler | **HIGH** - Different resampling strategy |
| **Duplicate Removal** | None | Removes absolute duplicates | **LOW** - May remove redundant features |
| **Temporal Features** | None | 3 temporal features added | **HIGH** - Adds leakage features |

---

## Feature Count Summary

### XGBoost Model
- Regular features: **51**
- Embedding features (PCA): **96**
- **Total: 147 features**

### Multi-Classifier Model
- Regular features: **51**
- Embedding features (PCA): **96**
- Temporal features: **3**
- **Total: 150 features** (before duplicate elimination)
- **Final: ~147-150 features** (after duplicate elimination)

---

## Key Differences

### 1. Temporal Features (CRITICAL)

**Multi-Classifier Model ONLY:**
- Adds 3 temporal features that use current date/time
- These are **data leakage features** (use future information)
- May significantly improve performance but are not valid for real-world deployment

**Impact:** ⚠️ **HIGH** - These features provide strong signal but represent temporal leakage

### 2. Feature Scaling Approach

**XGBoost:**
- Scales ALL features (including binary/one-hot)
- May distort binary feature distributions

**Multi-Classifier:**
- Scales ONLY numeric features
- Preserves binary/one-hot feature distributions
- More appropriate for mixed feature types

**Impact:** ✅ **MODERATE** - Multi-classifier approach is more correct

### 3. Class Imbalance Handling

**XGBoost:**
- SMOTETomek/SMOTEENN (synthetic oversampling + cleaning)
- Incremental processing (10k chunks) for OOM resistance
- More sophisticated resampling

**Multi-Classifier:**
- RandomUnderSampler (undersampling majority class)
- Simpler but faster approach
- May lose information from majority class

**Impact:** ⚠️ **HIGH** - XGBoost's SMOTE approach is generally superior

### 4. Duplicate Feature Elimination

**XGBoost:**
- No explicit duplicate removal
- Relies on model to handle redundant features

**Multi-Classifier:**
- Explicitly removes absolute duplicate features
- May reduce feature count slightly

**Impact:** ✅ **LOW** - Minor improvement in feature quality

---

## Recommendations

### For XGBoost Model
1. ✅ **Keep current approach** - Well-tested and robust
2. ⚠️ **Consider selective scaling** - Only scale numeric features
3. ✅ **Keep SMOTE** - Superior resampling approach

### For Multi-Classifier Model
1. ⚠️ **Remove temporal features** - If deploying to production (data leakage)
2. ✅ **Keep selective scaling** - Better approach for mixed features
3. ⚠️ **Consider SMOTE** - Instead of RandomUnderSampler for better performance
4. ✅ **Keep duplicate elimination** - Good practice

---

## Conclusion

### Feature Set Similarity: **98%**

Both models use:
- ✅ Same 51 regular features
- ✅ Same 96 embedding features (PCA-compressed)
- ✅ Same PCA compression (32 components per family)

### Key Differences:
1. **Multi-Classifier adds 3 temporal features** (+2% feature count)
2. **Different scaling approach** (selective vs. all)
3. **Different resampling** (SMOTE vs. RandomUnderSampler)
4. **Duplicate elimination** (multi-classifier only)

### Overall Assessment:
- **XGBoost**: More robust preprocessing (SMOTE), simpler feature set
- **Multi-Classifier**: Adds temporal leakage features (may boost performance but invalid for production), better scaling approach

**Recommendation:** If temporal features are acceptable (for competition/experimentation), multi-classifier model has slight advantage. For production, use XGBoost approach without temporal features.

