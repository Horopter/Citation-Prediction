"""
Utility functions to add missing features from reference implementation.

Features to add:
1. any_repository_has_fulltext (from open_access)
2. version features (fill nulls, one-hot encode)
3. Primary location source features (extract from primary_location)
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional
import ast


def safe_parse_json(value):
    """Safely parse JSON string or return empty dict/list."""
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value) if value else {}
        except:
            return {}
    return {}


def extract_any_repository_has_fulltext(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract any_repository_has_fulltext from open_access column.
    
    If open_access column exists, extract the feature.
    If is_oa already exists but any_repository_has_fulltext doesn't, try to extract it.
    """
    df_result = df.clone()
    
    # Check if open_access column exists (raw data)
    if 'open_access' in df_result.columns:
        open_access_parsed = df_result.select('open_access').to_series().map_elements(
            safe_parse_json, return_dtype=pl.Object
        )
        
        any_repository_has_fulltext_values = []
        for oa in open_access_parsed:
            if isinstance(oa, dict):
                any_repository_has_fulltext_values.append(
                    1.0 if oa.get('any_repository_has_fulltext', False) else 0.0
                )
            else:
                any_repository_has_fulltext_values.append(0.0)
        
        df_result = df_result.with_columns(
            pl.Series('any_repository_has_fulltext', any_repository_has_fulltext_values, dtype=pl.Float32)
        )
        print("  ✅ Extracted 'any_repository_has_fulltext' from open_access")
    elif 'any_repository_has_fulltext' not in df_result.columns:
        # If open_access doesn't exist and feature is missing, fill with 0
        df_result = df_result.with_columns(
            pl.lit(0.0).alias('any_repository_has_fulltext').cast(pl.Float32)
        )
        print("  ⚠️ 'open_access' column not found, filled 'any_repository_has_fulltext' with 0")
    
    return df_result


def extract_version_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract version features: fill nulls with 'noVersion' and one-hot encode.
    
    Returns dataframe with version features added and original version column dropped.
    """
    df_result = df.clone()
    
    if 'version' in df_result.columns:
        # Fill nulls with 'noVersion'
        df_result = df_result.with_columns(
            pl.col('version').fill_null('noVersion').alias('version_filled')
        )
        
        # Get unique version values
        version_values = df_result.select('version_filled').to_series().unique().to_list()
        
        # One-hot encode version
        for version_val in version_values:
            if version_val is not None:
                col_name = f"version_{version_val}"
                df_result = df_result.with_columns(
                    (pl.col('version_filled') == version_val).cast(pl.Float32).alias(col_name)
                )
        
        # Drop original and filled version columns
        df_result = df_result.drop(['version', 'version_filled'])
        print(f"  ✅ One-hot encoded 'version' ({len(version_values)} categories)")
    else:
        # If version doesn't exist, create a default 'version_noVersion' column
        df_result = df_result.with_columns(
            pl.lit(1.0).alias('version_noVersion').cast(pl.Float32)
        )
        print("  ⚠️ 'version' column not found, created default 'version_noVersion'")
    
    return df_result


def extract_primary_location_source_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract source fields from primary_location column.
    
    Extracts all source fields from primary_location.source dict,
    then drops specific columns as reference implementation does.
    
    Columns to drop (matching reference):
    - pdf_url, license, license_id
    - source.issn_l, source.issn
    - source.host_organization, source.host_organization_name
    - source.is_oa (we already have is_oa_venue)
    - source.id, source.display_name
    - source.is_in_doaj (we already have is_in_doaj)
    - source.is_indexed_in_scopus
    - source.is_core
    - source.host_organization_lineage, source.host_organization_lineage_names
    - source.type
    """
    df_result = df.clone()
    
    if 'primary_location' in df_result.columns:
        primary_location_parsed = df_result.select('primary_location').to_series().map_elements(
            safe_parse_json, return_dtype=pl.Object
        )
        
        # Collect all source fields
        source_fields_dict = {}
        sample_size = min(1000, len(primary_location_parsed))  # Sample to find all keys
        
        for ploc in primary_location_parsed[:sample_size]:
            if isinstance(ploc, dict) and 'source' in ploc:
                source = ploc['source']
                if isinstance(source, dict):
                    for key in source.keys():
                        if key not in source_fields_dict:
                            source_fields_dict[key] = []
        
        # Extract all source fields for all rows
        for key in source_fields_dict.keys():
            values = []
            for ploc in primary_location_parsed:
                if isinstance(ploc, dict) and 'source' in ploc:
                    source = ploc['source']
                    if isinstance(source, dict) and key in source:
                        val = source[key]
                        # Convert to string for categorical, keep numeric as is
                        if isinstance(val, (int, float)):
                            values.append(float(val) if val is not None else 0.0)
                        else:
                            values.append(str(val) if val is not None else '')
                    else:
                        values.append(0.0 if isinstance(key, (int, float)) else '')
                else:
                    values.append(0.0 if isinstance(key, (int, float)) else '')
            
            # Create column name
            col_name = f"source_{key}"
            
            # Determine dtype
            if all(isinstance(v, (int, float)) for v in values[:100] if v != ''):
                df_result = df_result.with_columns(
                    pl.Series(col_name, values, dtype=pl.Float32)
                )
            else:
                # Keep as string for now (can be one-hot encoded later if needed)
                df_result = df_result.with_columns(
                    pl.Series(col_name, values, dtype=pl.Utf8)
                )
        
        # Drop columns as reference implementation does
        columns_to_drop = [
            'pdf_url', 'license', 'license_id',
            'source_issn_l', 'source_issn',
            'source_host_organization', 'source_host_organization_name',
            'source_is_oa',  # We already have is_oa_venue
            'source_id', 'source_display_name',
            'source_is_in_doaj',  # We already have is_in_doaj
            'source_is_indexed_in_scopus',
            'source_is_core',
            'source_host_organization_lineage', 'source_host_organization_lineage_names',
            'source_type'
        ]
        
        for col in columns_to_drop:
            if col in df_result.columns:
                df_result = df_result.drop(col)
        
        # Drop original primary_location column
        df_result = df_result.drop('primary_location')
        
        kept_cols = [c for c in df_result.columns if c.startswith('source_')]
        print(f"  ✅ Extracted primary location source features ({len(kept_cols)} features kept)")
        if kept_cols:
            print(f"     Kept: {kept_cols[:5]}{'...' if len(kept_cols) > 5 else ''}")
    else:
        print("  ⚠️ 'primary_location' column not found, skipping source feature extraction")
    
    return df_result


def add_reference_features(df: pl.DataFrame, 
                          add_any_repository: bool = True,
                          add_version: bool = True,
                          add_primary_location: bool = True) -> pl.DataFrame:
    """
    Add all missing features from reference implementation.
    
    Args:
        df: Input dataframe
        add_any_repository: Whether to add any_repository_has_fulltext
        add_version: Whether to add version features
        add_primary_location: Whether to add primary location source features
    
    Returns:
        Dataframe with added features
    """
    df_result = df.clone()
    
    print("\n🔧 Adding reference implementation features...")
    
    if add_any_repository:
        df_result = extract_any_repository_has_fulltext(df_result)
    
    if add_version:
        df_result = extract_version_features(df_result)
    
    if add_primary_location:
        df_result = extract_primary_location_source_features(df_result)
    
    print(f"  ✅ Feature engineering complete. Final shape: {df_result.shape}")
    
    return df_result

