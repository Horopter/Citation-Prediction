# Feature Set Comparison: Reference Implementation vs Our Implementation

## Overview

This document provides a detailed comparison of features used in a reference implementation versus our implementation (`model_multi_classifier_leakage_test.ipynb`).

---

## Feature Categories

### 1. Temporal/Date Features (Data Leakage Features)

**Reference Implementation:**
- ✅ `num_years_after_publication` = 2025 - publication_year
- ✅ `days_since_updated` = (datetime.today() - updated_date).days
- ✅ `days_since_publication` = (datetime.today() - publication_date).days

**Our Implementation:**
- ✅ `num_years_after_publication` = 2025 - publication_year (hardcoded 2025)
- ✅ `days_since_updated` = (datetime.now() - updated_date).total_days()
- ✅ `days_since_publication` = (datetime.now() - publication_date).total_days()

**Status:** ✅ **MATCH** - All temporal features present

---

### 2. Open Access Features

**Reference Implementation:**
- ✅ `is_oa` (extracted from open_access dict)
- ✅ `oa_status` (extracted, then one-hot encoded)
- ✅ `any_repository_has_fulltext` (extracted from open_access dict)
- ❌ `oa_url` (dropped after extraction)

**Our Implementation:**
- ✅ `is_oa` (extracted from open_access)
- ✅ `oa_status` (extracted, then one-hot encoded)
- ✅ `any_repository_has_fulltext` (extracted from open_access)
- ⚠️ `oa_url` (not explicitly handled, may not exist in our data)

**Status:** ✅ **MATCH** - Core features match

---

### 3. Author Features

**Reference Implementation:**
- ✅ `num_authorships` = count of author positions in authorships list

**Our Implementation:**
- ✅ `num_authorships` = count of author positions (matching logic)

**Status:** ✅ **MATCH**

---

### 4. Related Works & Grants

**Reference Implementation:**
- ✅ `num_related_words` = len(related_works list)
- ✅ `num_grants` = len(grants list)

**Our Implementation:**
- ✅ `num_related_words` = len(related_works list)
- ✅ `num_grants` = len(grants list)

**Status:** ✅ **MATCH**

---

### 5. Language Features

**Reference Implementation:**
- ✅ `new_language` = language.fillna('unknown')
- ✅ One-hot encoded: `new_language_{lang}` for each language

**Our Implementation:**
- ✅ `new_language` = language.fill_null('unknown')
- ✅ One-hot encoded: `new_language_{lang}` for each language

**Status:** ✅ **MATCH**

---

### 6. Version Features

**Reference Implementation:**
- ✅ `version` = version.fillna('noVersion')
- ✅ One-hot encoded: `version_{version_value}`

**Our Implementation:**
- ❌ **MISSING** - No version column handling

**Status:** ❌ **MISSING** - Version features not implemented

---

### 7. Primary Location / Source Features

**Reference Implementation:**
- ✅ Extracts ALL source fields via `pd.json_normalize(primary_location)`
- ✅ Then drops specific columns:
  - `pdf_url`, `license`, `license_id`
  - `source.issn_l`, `source.issn`
  - `source.host_organization`, `source.host_organization_name`
  - `source.is_oa`
  - `source.id`, `source.display_name`
  - `source.is_in_doaj`, `source.is_indexed_in_scopus`
  - `source.is_core`
  - `source.host_organization_lineage`, `source.host_organization_lineage_names`
  - `source.type`
- ⚠️ **Unknown**: Which source fields are kept (if any)

**Our Implementation:**
- ❌ **MISSING** - Drops `primary_location` entirely
- ❌ No source/venue features extracted

**Status:** ❌ **CRITICAL MISSING** - All venue/source information lost

---

### 8. Columns Dropped

**Reference Implementation Drops:**
- ✅ `abstract` (45% missing)
- ✅ `id`, `new_ids` (after extracting work_id)
- ✅ `doi_url` (19.5% missing)
- ✅ `ids` column
- ✅ `locations` (entire column)
- ✅ `title`
- ✅ `concepts`
- ✅ `type`, `type_crossref` (after processing)
- ✅ `publication_year`, `updated_date`, `publication_date` (after creating temporal features)
- ✅ `landing_page_url`
- ✅ `index` (if present)

**Our Implementation Drops:**
- ✅ `abstract`
- ✅ `id` (after extracting work_id)
- ✅ `doi_url`
- ✅ `ids`
- ✅ `locations`
- ✅ `title`
- ✅ `concepts`
- ✅ `type` (keeps type_crossref if present)
- ✅ `publication_year`, `updated_date`, `publication_date` (after creating temporal features)
- ⚠️ `landing_page_url` (not explicitly handled)
- ⚠️ `index` (not explicitly handled)

**Status:** ✅ **MOSTLY MATCH** - Minor differences in explicit handling

---

### 9. Boolean Flag Features

**Reference Implementation:**
- ✅ `is_oa` (float)
- ✅ `any_repository_has_fulltext` (float)
- ✅ `is_accepted` (float, if present)
- ✅ `is_published` (float, if present)

**Our Implementation:**
- ✅ `is_oa` (Float32)
- ✅ `any_repository_has_fulltext` (Float32)
- ⚠️ `is_accepted` (not explicitly handled)
- ⚠️ `is_published` (not explicitly handled)

**Status:** ⚠️ **PARTIAL MATCH** - Core flags match, some optional flags missing

---

### 10. Embedding Features

**Reference Implementation:**
- ❌ **NO EMBEDDINGS** - Uses only base features

**Our Implementation:**
- ✅ **Sentence Transformers** embeddings (384 dims → 32 PCA components)
- ✅ **SciBERT** embeddings (768 dims → 32 PCA components)
- ✅ **SPECTER2** embeddings (768 dims → 32 PCA components)
- ✅ Total: ~96 embedding features (3 families × 32 PCA components)

**Status:** ✅ **EXTRA FEATURES** - We have embeddings, reference implementation doesn't

---

### 11. Regular Features from data_exploration_organized.ipynb

**Our Implementation (XGBoost Features - 51 actual feature names):**

**Basic Metadata (3 features):**
- ✅ `publication_year`
- ✅ `type`
- ✅ `language`

**Text Features (5 features):**
- ✅ `title_length`
- ✅ `abstract_length`
- ✅ `has_abstract`
- ✅ `title_word_count`
- ✅ `abstract_word_count`

**Author Features (10 features):**
- ✅ `num_authors`
- ✅ `num_institutions`
- ✅ `first_author_h_index`
- ✅ `first_author_citations`
- ✅ `first_author_papers`
- ✅ `max_author_h_index`
- ✅ `avg_author_h_index`
- ✅ `total_author_citations`
- ✅ `avg_author_citations`

**Venue Features (5 features):**
- ✅ `venue_impact_factor`
- ✅ `venue_h_index`
- ✅ `venue_citations`
- ✅ `is_oa_venue`
- ✅ `is_in_doaj`

**Concept Features (4 features):**
- ✅ `num_concepts`
- ✅ `max_concept_score`
- ✅ `avg_concept_score`
- ✅ `num_high_concepts`

**Open Access Features (1 + one-hot):**
- ✅ `is_oa`
- ✅ `oa_status_{oa_status}` (one-hot encoded)

**Reference Features (2 features):**
- ✅ `num_references`
- ✅ `num_related_works`

**Identifier Features (3 features):**
- ✅ `has_doi`
- ✅ `has_pmid`
- ✅ `has_pmcid`

**Grant Features (2 features):**
- ✅ `num_grants`
- ✅ `has_grants`

**Location Features (1 feature):**
- ✅ `num_locations`

**Topic Features (3 features):**
- ✅ `num_topics`
- ✅ `max_topic_score`
- ✅ `avg_topic_score`

**Keyword Features (1 feature):**
- ✅ `num_keywords`

**Mesh Terms (1 feature):**
- ✅ `num_mesh_terms`

**SDG Features (2 features):**
- ✅ `num_sdgs`
- ✅ `has_sdgs`

**NLP Features (8 features):**
- ✅ `nlp_char_count`
- ✅ `nlp_word_count`
- ✅ `nlp_avg_word_length`
- ✅ `nlp_sentence_count`
- ✅ `nlp_avg_sentence_length`
- ✅ `nlp_capital_ratio`
- ✅ `nlp_number_ratio`
- ✅ `nlp_punctuation_ratio`

**Total Regular Features: 51 features** (excluding one-hot encoded `oa_status_*`)

**Reference Implementation:**
- ⚠️ **Unknown exact count** - Uses base CSV features only
- Likely similar base features but extracted differently
- Estimated ~20-30 base features

**Status:** ✅ **COMPREHENSIVE** - We have 51 regular features vs reference implementation's ~20-30

---

## Feature Count Summary

### Reference Implementation (Estimated)
- Base features: ~20-30 (from CSV columns)
- Temporal features: 3
- Open access features: 1 + one-hot(oa_status)
- Language features: one-hot(new_language)
- Version features: one-hot(version)
- Author features: 1
- Related works/grants: 2
- **Total: ~30-50 features** (excluding one-hot expansions)

### Our Implementation
- Regular features: **51 features** (from data_exploration_organized.ipynb)
  - Basic metadata: 3
  - Text features: 5
  - Author features: 10
  - Venue features: 5
  - Concept features: 4
  - Reference features: 2
  - Identifier features: 3
  - Grant features: 2
  - Location features: 1
  - Topic features: 3
  - Keyword/Mesh/SDG: 3
  - NLP features: 8
- Temporal features: 3 (`num_years_after_publication`, `days_since_updated`, `days_since_publication`)
- Open access features: 1 (`is_oa`) + one-hot(`oa_status_*`)
- Language features: one-hot(`new_language_*`)
- Author count feature: 1 (`num_authorships` - from reference style)
- Related works/grants: 2 (`num_related_words`, `num_grants` - from reference style)
- Embedding features: **96 features** (3 families × 32 PCA components)
  - `sent_transformer_pca_0` through `sent_transformer_pca_31` (32 features)
  - `scibert_pca_0` through `scibert_pca_31` (32 features)
  - `specter2_pca_0` through `specter2_pca_31` (32 features)
- **Total: ~150-200 features** (excluding one-hot expansions)

---

## Critical Differences

### ❌ Missing in Our Implementation

1. **Version Features** (MODERATE)
   - Reference: `version` column → one-hot encoded
   - Our: Not implemented
   - Impact: May lose publication version information

2. **Primary Location / Source Features** (CRITICAL)
   - Reference: Extracts source fields, drops specific ones
   - Our: Drops entire `primary_location` column
   - Impact: **Loses all venue/source information** - could be highly predictive

3. **Some Boolean Flags** (LOW)
   - Reference: Explicitly handles `is_accepted`, `is_published`
   - Our: Not explicitly handled
   - Impact: Minor, may not exist in our data

### ✅ Extra in Our Implementation

1. **Embedding Features** (SIGNIFICANT)
   - Our: 96 embedding features (PCA-compressed)
   - Reference: None
   - Impact: **Major advantage** - semantic text information

2. **More Regular Features** (MODERATE)
   - Our: **51 regular features** from comprehensive extraction (see list above)
   - Reference: ~20-30 base features
   - Impact: More comprehensive feature set including:
     - Author statistics (h-index, citations, papers)
     - Venue metrics (impact factor, h-index, citations)
     - Concept/topic scores and statistics
     - NLP text statistics (8 features)
     - SDG and Mesh term features

---

## Feature Engineering Differences

### Data Preprocessing

**Reference Implementation:**
```python
# Concatenate train+val
data = pd.concat([train, val])
# Drop nulls/duplicates BEFORE feature engineering
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
# Then feature engineering
```

**Our Implementation:**
```python
# Load separately
X_train = load_base_features("train")
X_val = load_base_features("val")
# Feature engineering separately
# No explicit null/duplicate dropping
```

### Scaling

**Reference Implementation:**
- Uses explicit `num_cols` list
- Scales only columns in `num_cols`

**Our Implementation:**
- Dynamically identifies numeric columns
- Scales all numeric (non-binary) columns

---

## Recommendations

### High Priority

1. **Add Version Features**
   ```python
   if 'version' in df.columns:
       df['version'] = df['version'].fill_null('noVersion')
       # One-hot encode version
   ```

2. **Extract Primary Location Features**
   ```python
   # Extract source fields from primary_location
   # Drop specific columns as reference implementation does
   # Keep remaining source features
   ```

### Medium Priority

3. **Match Scaling Approach**
   - Use explicit column list instead of dynamic identification
   - Ensure same columns are scaled as reference implementation

4. **Handle Boolean Flags**
   - Explicitly handle `is_accepted`, `is_published` if present

### Low Priority

5. **Explicit Drop Handling**
   - Explicitly handle `landing_page_url`, `index` drops

---

## Summary

### ✅ What We Have That Reference Implementation Doesn't
- **Embedding features** (96 features) - Major advantage
- **More comprehensive regular features** (54+ vs ~20-30)

### ❌ What Reference Implementation Has That We Don't
- **Version features** - Moderate impact
- **Primary location/source features** - **CRITICAL** - High impact
- **Explicit boolean flag handling** - Low impact

### Overall Assessment

**Feature Count:** We have **more features** (~150-200 vs ~30-50)

**Feature Quality:** 
- ✅ We have embeddings (major advantage)
- ❌ We lose venue/source information (major disadvantage)
- ⚠️ Missing version features (moderate disadvantage)

**Recommendation:** Add version and primary_location features to match reference implementation while keeping our embedding advantage.

