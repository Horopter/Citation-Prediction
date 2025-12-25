#!/usr/bin/env python3
"""
Convert JSONL files to Parquet format using Polars
Memory-efficient version that processes in chunks
"""

import polars as pl
import json
from pathlib import Path
import tempfile
import shutil

def jsonl_to_parquet(jsonl_path, parquet_path, chunk_size=5000):
    """
    Convert JSONL file to Parquet format in chunks to handle large files
    Uses temporary files to avoid memory issues
    Handles schema inconsistencies by inferring schema from sample first
    """
    if Path(parquet_path).exists():
        print(f"{parquet_path} already exists, skipping conversion...")
        return
    
    print(f"Converting {jsonl_path} to {parquet_path}...")
    
    # First, sample records to infer schema
    print("  Inferring schema from sample...")
    sample_records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Sample first 1000 records
                break
            try:
                data = json.loads(line.strip())
                sample_records.append(data)
            except json.JSONDecodeError:
                continue
    
    if not sample_records:
        print("  Error: Could not read any records from file!")
        return
    
    # Infer schema from sample
    sample_df = pl.DataFrame(sample_records)
    inferred_schema = sample_df.schema
    print(f"  Inferred schema with {len(inferred_schema)} columns")
    
    # Create temporary directory for chunk files
    temp_dir = Path(tempfile.mkdtemp())
    chunk_files = []
    
    try:
        current_chunk = []
        total_rows = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    current_chunk.append(data)
                    
                    if len(current_chunk) >= chunk_size:
                        # Create DataFrame with inferred schema to handle missing fields
                        df_chunk = pl.DataFrame(current_chunk, schema_overrides=inferred_schema)
                        chunk_file = temp_dir / f"chunk_{len(chunk_files)}.parquet"
                        df_chunk.write_parquet(str(chunk_file), compression='zstd')
                        chunk_files.append(chunk_file)
                        current_chunk = []
                        total_rows += len(df_chunk)
                        
                        if (i + 1) % 50000 == 0:
                            print(f"  Processed {i+1} lines ({total_rows} rows written)...")
                except json.JSONDecodeError as e:
                    print(f"\nWarning: Skipping line {i+1} due to JSON error: {e}")
                    continue
                except Exception as e:
                    print(f"\nWarning: Error processing line {i+1}: {e}")
                    continue
        
        # Write remaining chunk
        if current_chunk:
            df_chunk = pl.DataFrame(current_chunk, schema_overrides=inferred_schema)
            chunk_file = temp_dir / f"chunk_{len(chunk_files)}.parquet"
            df_chunk.write_parquet(str(chunk_file), compression='zstd')
            chunk_files.append(chunk_file)
            total_rows += len(df_chunk)
        
        print(f"  Processed {total_rows} total rows in {len(chunk_files)} chunks")
        
        # Read and concatenate all chunks
        print("  Concatenating chunks...")
        if chunk_files:
            # Read all chunk files and concatenate with diagonal strategy to handle schema differences
            dfs = [pl.read_parquet(str(f)) for f in chunk_files]
            # Use 'diagonal' to handle any remaining schema differences
            df = pl.concat(dfs, how='diagonal')
            
            print(f"  Total rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            
            # Write final parquet file
            df.write_parquet(parquet_path, compression='zstd')
            print(f"  Saved to {parquet_path}")
            print(f"  File size: {Path(parquet_path).stat().st_size / (1024**3):.2f} GB")
        else:
            print("  No data to convert!")
    
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # Determine data directories (organized structure)
    # Check for organized structure first (src/data/raw and src/data/processed)
    data_raw_dir = Path('data/raw')
    data_processed_dir = Path('data/processed')
    
    # Fallback to old structure for compatibility
    if not data_raw_dir.exists():
        data_raw_dir = Path('data')
        data_processed_dir = Path('data')
    
    if not data_raw_dir.exists():
        data_raw_dir = Path('data')
        data_processed_dir = Path('data')
    
    if not data_raw_dir.exists():
        data_raw_dir = Path('.')
        data_processed_dir = Path('.')
    
    # Ensure processed directory exists
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert all JSONL files to Parquet
    # Handle both test.no_label.jsonl and test.jsonl
    files = [
        (data_raw_dir / 'train.jsonl', data_processed_dir / 'train.parquet'),
        (data_raw_dir / 'val.jsonl', data_processed_dir / 'val.parquet'),
    ]
    
    # Check for test file (either test.no_label.jsonl or test.jsonl)
    test_file = None
    if (data_raw_dir / 'test.no_label.jsonl').exists():
        test_file = (data_raw_dir / 'test.no_label.jsonl', data_processed_dir / 'test.parquet')
    elif (data_raw_dir / 'test.jsonl').exists():
        test_file = (data_raw_dir / 'test.jsonl', data_processed_dir / 'test.parquet')
    
    if test_file:
        files.append(test_file)
    
    print(f"Converting JSONL files from: {data_raw_dir}")
    print(f"Output parquet files to: {data_processed_dir}")
    for jsonl_file, parquet_file in files:
        if jsonl_file.exists():
            jsonl_to_parquet(str(jsonl_file), str(parquet_file))
        else:
            print(f"Warning: {jsonl_file} not found, skipping...")

