# Derived Features Comparison: Reference Implementation vs Our Implementation

## Overview

This document compares **derived features** (features created/engineered from raw data) between a reference implementation and our implementation.

---

## Reference Implementation Derived Features

Based on the reference implementation, the following derived features are created:

### 1. Temporal Features (Date-Derived)
- ✅ `num_years_after_publication` = 2025 - publication_year
- ✅ `days_since_updated` = (datetime.today() - updated_date).days
- ✅ `days_since_publication` = (datetime.today() - publication_date).days

### 2. Open Access Features (Extracted from open_access dict)
- ✅ `is_oa` = extracted from open_access['is_oa']
- ✅ `oa_status` = extracted from open_access['oa_status'] (then one-hot encoded)
- ✅ `any_repository_has_fulltext` = extracted from open_access['any_repository_has_fulltext']
- ❌ `oa_url` = extracted then dropped

### 3. Author Features
- ✅ `num_authorships` = count of author positions in authorships list

### 4. Related Works Features
- ✅ `num_related_words` = len(related_works list)

### 5. Grant Features
- ✅ `num_grants` = len(grants list)

### 6. Language Features
- ✅ `new_language` = language.fillna('unknown') (then one-hot encoded)

### 7. Version Features
- ✅ `version` = version.fillna('noVersion') (then one-hot encoded)

### 8. Primary Location Features
- ✅ Extracts source fields from primary_location via json_normalize
- ✅ Drops specific columns but may keep some source features

---

## Our Implementation Derived Features

### Currently Created (in add_temporal_features function)
- ✅ `num_years_after_publication` = 2025 - publication_year
- ✅ `days_since_updated` = (datetime.now() - updated_date).total_days()
- ✅ `days_since_publication` = (datetime.now() - publication_date).total_days()

### Already Present in XGBoost Features (from data_exploration_organized.ipynb)
- ✅ `is_oa` (extracted from open_access)
- ✅ `oa_status_{status}` (one-hot encoded directly)
- ✅ `num_grants` (count of grants)
- ✅ `num_related_works` (count of related works - note: different name from reference's `num_related_words`)
- ✅ `num_authors` (count of authors - note: different from reference's `num_authorships`)
- ✅ `language` (original language column)

### Missing Derived Features

1. ❌ **`any_repository_has_fulltext`** - Not extracted from open_access
2. ❌ **`num_authorships`** - We have `num_authors` but reference counts author positions differently
3. ❌ **`num_related_words`** - We have `num_related_works` (same data, different name)
4. ❌ **`new_language`** - We have `language` but reference creates `new_language` with fillna('unknown')
5. ❌ **`version`** - Not handled at all
6. ❌ **Primary location source features** - Not extracted

---

## Detailed Comparison

### ✅ Present in Our Code

| Reference Feature | Our Implementation | Status |
|---------------------|-------------------|--------|
| `num_years_after_publication` | ✅ `num_years_after_publication` | ✅ MATCH |
| `days_since_updated` | ✅ `days_since_updated` | ✅ MATCH |
| `days_since_publication` | ✅ `days_since_publication` | ✅ MATCH |
| `is_oa` | ✅ `is_oa` (from XGBoost) | ✅ MATCH |
| `oa_status` (one-hot) | ✅ `oa_status_{status}` (from XGBoost) | ✅ MATCH |
| `num_grants` | ✅ `num_grants` (from XGBoost) | ✅ MATCH |

### ⚠️ Different Implementation

| Reference Feature | Our Implementation | Difference |
|---------------------|-------------------|------------|
| `num_authorships` | `num_authors` | Reference counts author positions, we count authors |
| `num_related_words` | `num_related_works` | Same data, different name |
| `new_language` | `language` | Reference fills nulls with 'unknown', we keep original |

### ❌ Missing in Our Code

| Reference Feature | Impact | Priority |
|---------------------|--------|----------|
| `any_repository_has_fulltext` | MODERATE | Medium |
| `version` (one-hot) | MODERATE | Medium |
| Primary location source features | CRITICAL | High |

---

## Summary

### ✅ Present: 6/11 derived features
- All temporal features (3)
- Core open access features (2)
- Grant count (1)

### ⚠️ Different: 3/11 derived features
- Author count (different counting method)
- Related works (different name)
- Language (different handling)

### ❌ Missing: 2/11 critical derived features
- `any_repository_has_fulltext`
- `version` features
- Primary location source features (CRITICAL)

---

## Recommendations

### High Priority
1. **Add `any_repository_has_fulltext`** extraction from open_access
2. **Add `version`** handling (fill nulls, one-hot encode)
3. **Extract primary_location source features** (even if we drop most, some may be kept)

### Medium Priority
4. **Add `num_authorships`** (count author positions) - may be redundant with `num_authors`
5. **Rename `num_related_works` to `num_related_words`** for consistency (or add alias)
6. **Create `new_language`** with fill_null('unknown') for consistency

---

## Current Implementation Status

### What's Actually Being Used

**Current Pipeline Flow:**
1. Loads XGBoost features (base features + embeddings)
2. **Only adds temporal features** via `add_temporal_features()` function
3. Combines regular + embeddings (PCA) + temporal features

**Note:** The notebook contains `extract_features_reference_style()` function that creates many derived features, but **this function is NOT being called** in the current pipeline. It references `X_train_df` which doesn't exist - the current flow uses `X_train_combined`.

### Derived Features Actually Created

**Currently Created (Only Temporal):**
- ✅ `num_years_after_publication`
- ✅ `days_since_updated`
- ✅ `days_since_publication`

**From XGBoost Features (Already Present):**
- ✅ `is_oa` (from data_exploration_organized.ipynb)
- ✅ `oa_status_{status}` (one-hot encoded)
- ✅ `num_grants`
- ✅ `num_related_works` (note: different name from reference's `num_related_words`)
- ✅ `num_authors` (note: different from reference's `num_authorships`)

### Missing Derived Features

**Not Created in Current Pipeline:**
1. ❌ `any_repository_has_fulltext` - Not extracted from open_access
2. ❌ `num_authorships` - We have `num_authors` but counting method differs
3. ❌ `num_related_words` - We have `num_related_works` (same data, different name)
4. ❌ `new_language` - We have `language` but reference creates `new_language` with fillna('unknown')
5. ❌ `version` - Not handled at all
6. ❌ Primary location source features - Not extracted

---

## Conclusion

**Answer: NO** - Not all reference implementation derived features are present in our code.

**Currently Created:**
- ✅ All temporal features (3 features)
- ✅ Core features from XGBoost pipeline (51 features)

**Missing Reference Implementation Derived Features:**
- ❌ `any_repository_has_fulltext` (moderate impact)
- ❌ `version` features (moderate impact)
- ❌ Primary location source features (critical impact)
- ❌ `num_authorships` (different counting method)
- ❌ `num_related_words` (different name)
- ❌ `new_language` (different handling)

**Note:** The notebook contains code (`extract_features_reference_style`) that would create these features, but it's **not being executed** in the current pipeline. The current implementation only adds temporal features to XGBoost features.

