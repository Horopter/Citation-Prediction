# Critical Analysis: Differences Between Reference Implementation and Our Implementation

## Overview

This document provides a critical analysis of differences between a reference implementation and our implementation (`model_multi_classifier_leakage_test.ipynb`).

---

## 🔴 Critical Differences

### 1. Data Preprocessing Order

**Reference Implementation:**
```python
# Concatenate train and val FIRST
data = pd.concat([train, val], axis=0, ignore_index=True)

# Drop nulls and duplicates BEFORE feature engineering
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Then do feature engineering on combined dataset
```

**Our Implementation:**
```python
# Load train/val/test separately
X_train_df = load_base_features("train")
X_val_df = load_base_features("val")
X_test_df = load_base_features("test")

# Do feature engineering separately on each split
# No null/duplicate dropping before feature engineering
```

**Impact**: 
- **HIGH**: Reference implementation processes ~71.7% of data after dropping nulls/duplicates
- Our approach may include nulls/duplicates that could affect feature engineering
- Feature distributions may differ between train/val if nulls are handled differently

---

### 2. Primary Location Normalization

**Reference Implementation:**
```python
# Uses pd.json_normalize to extract ALL source fields
test['primary_location'] = test['primary_location'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
dict_df = pd.json_normalize(test['primary_location'])
test = pd.concat([test.drop(columns=['primary_location']), dict_df], axis=1)

# Then drops specific columns
cols_to_remove = ['pdf_url','license','license_id','source.issn_l','source.issn',
                  'source.host_organization','source.host_organization_name','source.is_oa']
src_cols = ['source.id','source.display_name','source.is_in_doaj','source.is_indexed_in_scopus',
            'source.is_core','source.host_organization_lineage','source.host_organization_lineage_names','source.type']
```

**Our Implementation:**
```python
# Drops primary_location entirely
if 'primary_location' in df_processed.columns:
    df_processed = df_processed.drop('primary_location')
    print("  ✅ Processed 'primary_location' (dropped nested structure)")
```

**Impact**: 
- **CRITICAL**: Reference implementation extracts potentially valuable source features (venue information)
- We lose all primary_location information
- This could significantly impact model performance if venue features are predictive

---

### 3. Version Column Handling

**Reference Implementation:**
```python
# Fills version column with 'noVersion' if missing
if 'version' in test.columns:
    test['version'].fillna('noVersion', inplace=True)

# Then includes version in categorical dummies
cat_columns = ['new_language','oa_status','version']
test = pd.get_dummies(data=test, columns=present_cat_columns, prefix=present_cat_columns)
```

**Our Implementation:**
```python
# No version column handling
# Version is not mentioned in our feature engineering
```

**Impact**: 
- **MODERATE**: Missing version information could be informative
- Version might correlate with publication status or quality

---

### 4. Scaling Approach

**Reference Implementation:**
```python
# Uses explicit list of numeric columns to scale
num_cols = [...]  # Explicit list defined during training
test[pres] = scaler.transform(test[pres])  # Only scales columns in num_cols
```

**Our Implementation:**
```python
# Dynamically identifies numeric columns
for col in feature_column_names:
    col_data = X_train_engineered.select(col).to_series()
    unique_vals = col_data.unique().to_list()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        binary_cols.append(col)
    else:
        if not col.startswith('new_language_') and not col.startswith('oa_status_'):
            numeric_cols_to_scale.append(col)
```

**Impact**: 
- **MODERATE**: Different columns may be scaled
- Reference implementation's explicit list ensures consistency
- Our dynamic approach may scale columns that shouldn't be scaled (or miss some that should)

---

### 5. Categorical Encoding Method

**Reference Implementation:**
```python
# Uses pandas get_dummies
cat_columns = ['new_language','oa_status','version']
test = pd.get_dummies(data=test, columns=present_cat_columns, prefix=present_cat_columns).astype('float')
```

**Our Implementation:**
```python
# Manual one-hot encoding (Polars doesn't have get_dummies)
if 'new_language' in df_processed.columns:
    lang_values = df_processed.select('new_language').to_series().unique().to_list()
    for lang in lang_values:
        if lang is not None:
            col_name = f"new_language_{lang}"
            df_processed = df_processed.with_columns(
                (pl.col('new_language') == lang).cast(pl.Float32).alias(col_name)
            )
```

**Impact**: 
- **LOW**: Should produce same result, but:
  - Reference implementation uses `prefix` parameter (e.g., `new_language_en`)
  - Our approach uses `f"new_language_{lang}"` (should match)
  - Potential difference if categories differ between train/test

---

### 6. ID Column Handling

**Reference Implementation:**
```python
# Extracts work_id from id, then drops both
data['new_ids'] = data['id'].str.split('.org/', expand=True)[1]
data.drop(['id','new_ids'], axis=1, inplace=True)
```

**Our Implementation:**
```python
# Stores id for later use, drops it during feature engineering
if 'id' in df_processed.columns:
    df_processed = df_processed.drop('id')
```

**Impact**: 
- **LOW**: Both drop id, but reference implementation extracts work_id first
- We extract work_id later for submission (different timing)

---

### 7. Date Feature Calculation

**Reference Implementation:**
```python
# Uses hardcoded year 2025
test['num_years_after_publication'] = 2025 - test['publication_year']

# Uses datetime.today() for date differences
test['days_since_updated'] = (datetime.today() - pd.to_datetime(test['updated_date'])).dt.days
test['days_since_publication'] = (datetime.today() - pd.to_datetime(test['publication_date'])).dt.days
```

**Our Implementation:**
```python
# Uses datetime.now().year (dynamic)
current_year = datetime.now().year
df_processed = df_processed.with_columns(
    (current_year - pl.col('publication_year')).alias('num_years_after_publication')
)

# Uses datetime.now() for date differences
today = datetime.now()
df_processed = df_processed.with_columns(
    ((today - pl.col('updated_date')).dt.total_days()).alias('days_since_updated')
)
```

**Impact**: 
- **LOW**: Reference implementation uses hardcoded 2025 (likely from when notebook was written)
- Our approach is dynamic (will use current year)
- This could cause slight differences in date features

---

### 8. Missing Column Handling in primary_location

**Reference Implementation:**
```python
# Explicitly drops many source.* columns
cols_to_remove = ['pdf_url','license','license_id','source.issn_l','source.issn',
                  'source.host_organization','source.host_organization_name','source.is_oa']
src_cols = ['source.id','source.display_name','source.is_in_doaj','source.is_indexed_in_scopus',
            'source.is_core','source.host_organization_lineage','source.host_organization_lineage_names','source.type']
for c in cols_to_remove + src_cols:
    test.drop([c], axis=1, inplace=True, errors='ignore')
test.pop('source') if 'source' in test.columns else None
```

**Our Implementation:**
```python
# Drops primary_location entirely, doesn't extract source fields
df_processed = df_processed.drop('primary_location')
```

**Impact**: 
- **CRITICAL**: We lose all source/venue information
- Reference implementation extracts source fields but then drops many of them
- We don't know which source fields reference implementation keeps (if any)

---

### 9. Data Library

**Reference Implementation:**
- Uses **pandas** (`pd.read_csv`, `pd.json_normalize`, `pd.get_dummies`)

**Our Implementation:**
- Uses **polars** (`pl.read_parquet`, manual JSON parsing, manual one-hot encoding)

**Impact**: 
- **LOW-MODERATE**: Different libraries, but should produce similar results
- Polars is faster but requires manual implementation of some pandas features
- Potential differences in handling edge cases

---

### 10. Feature Engineering Timing

**Reference Implementation:**
- Feature engineering happens on **combined train+val** dataset
- Then splits back (presumably) for training

**Our Implementation:**
- Feature engineering happens **separately** on train/val/test
- No concatenation step

**Impact**: 
- **MODERATE**: 
  - Reference implementation ensures consistent feature extraction across splits
  - Our approach may have inconsistencies if categories differ between splits
  - One-hot encoding may produce different columns if categories differ

---

## ⚠️ Moderate Differences

### 11. Open Access URL Handling

**Reference Implementation:**
```python
# Drops oa_url after json_normalize
test.drop(['oa_url'], axis=1, inplace=True, errors='ignore')
```

**Our Implementation:**
```python
# Doesn't explicitly handle oa_url (may not exist in our data structure)
```

**Impact**: LOW - Minor difference

---

### 12. Boolean Flag Handling

**Reference Implementation:**
```python
# Explicitly converts boolean flags to float
flag_cols = ['is_oa','any_repository_has_fulltext','is_accepted','is_published']
for c in flag_cols:
    if c in test.columns:
        test[c] = test[c].astype('float')
```

**Our Implementation:**
```python
# Creates is_oa and any_repository_has_fulltext as Float32 directly
pl.Series('is_oa', is_oa_values, dtype=pl.Float32)
pl.Series('any_repository_has_fulltext', any_repository_has_fulltext_values, dtype=pl.Float32)
```

**Impact**: LOW - Should be equivalent

---

## Summary of Critical Issues

### 🔴 Must Fix

1. **Primary Location**: We drop it entirely, reference implementation extracts source fields
   - **Action**: Implement json_normalize equivalent for primary_location
   - **Priority**: HIGH

2. **Data Preprocessing**: Reference implementation drops nulls/duplicates before feature engineering
   - **Action**: Add null/duplicate dropping step before feature engineering
   - **Priority**: HIGH

3. **Version Column**: Reference implementation handles it, we don't
   - **Action**: Add version column handling (fill nulls, one-hot encode)
   - **Priority**: MODERATE

### ⚠️ Should Fix

4. **Scaling**: Use explicit column list instead of dynamic identification
   - **Action**: Define explicit `num_cols` list matching reference implementation
   - **Priority**: MODERATE

5. **Date Features**: Use hardcoded year 2025 to match reference implementation
   - **Action**: Use `2025` instead of `datetime.now().year`
   - **Priority**: LOW

---

## Expected Impact on Performance

1. **Primary Location Loss**: Could reduce performance by 2-5% if venue features are predictive
2. **Null/Duplicate Handling**: Could affect feature distributions, impact depends on data quality
3. **Version Missing**: Likely minimal impact unless version correlates strongly with label
4. **Scaling Differences**: Could affect model performance if wrong columns are scaled

---

## Recommendations

### Immediate Actions

1. **Implement primary_location normalization**:
   ```python
   # Extract source fields using json_normalize equivalent
   # Drop specific columns as reference implementation does
   ```

2. **Add null/duplicate dropping**:
   ```python
   # Concatenate train+val, drop nulls/duplicates
   # Then do feature engineering
   ```

3. **Add version column handling**:
   ```python
   # Fill nulls with 'noVersion'
   # Include in categorical dummies
   ```

4. **Use explicit numeric columns list**:
   ```python
   # Define num_cols explicitly
   # Match reference implementation's scaling approach exactly
   ```

### Testing

- Compare feature counts between implementations
- Verify one-hot encoded columns match
- Check that scaled columns match reference implementation's `num_cols`

---

## Conclusion

The most critical differences are:
1. **Primary location handling** - We lose all venue information
2. **Data preprocessing order** - Different null/duplicate handling
3. **Version column** - Missing in our implementation

These differences could significantly impact model performance and feature consistency. The primary_location issue is the most critical as it represents a complete loss of potentially valuable features.

