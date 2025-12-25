# Creating Safe Zip Files

This project contains large files (data, models, virtual environments) that can trigger "zip bomb" detection errors when creating zip files. Use the provided script to create safe zip files.

## Quick Start

```bash
# Create a safe zip file (excludes large files)
./src/scripts/create_safe_zip.sh Kaggle2_safe.zip
```

## What Gets Excluded

The script automatically excludes:

- **Virtual environments**: `venv/` directory (1.5GB+)
- **NLTK data**: `data/nltk_data/` (can be regenerated)
- **Cache directories**: `.cache/`, `.pip-cache/`, etc.
- **Build artifacts**: `build/`, `dist/`, `*.egg-info/`
- **Logs**: `logs/*.log`, `logs/*.out`, `logs/*.err`
- **Temporary files**: `*.tmp`, `tmp/`, `temp/`
- **IDE files**: `.vscode/`, `.idea/`, `.DS_Store`
- **Jupyter checkpoints**: `.ipynb_checkpoints/`

## What Gets Included

**Important**: The following large directories are INCLUDED in the zip:

- **Data files**: `data/processed/`, `data/results/` (9.3GB+)
- **Model files**: `models/` directory (511MB+)

⚠ **Warning**: Including these will create a very large zip file (10GB+). Ensure you have sufficient disk space and bandwidth.

## Manual Zip Creation

If you prefer to create zip files manually, exclude these patterns:

```bash
zip -r Kaggle2.zip . \
    -x "venv/*" \
    -x "data/nltk_data/*" \
    -x ".cache/*" \
    -x ".pip-cache/*" \
    -x "__pycache__/*" \
    -x "*.pyc" \
    -x ".ipynb_checkpoints/*" \
    -x "logs/*.log" \
    -x "*.zip"
```

Note: This includes `data/` and `models/` folders, so the zip will be large (10GB+).

## Restoring Excluded Files

After extracting the zip file, you'll need to:

1. **Recreate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download NLTK data** (if needed):
   ```bash
   python src/scripts/download_nltk_data.py
   ```

Note: Data files and models are already included in the zip, so no additional download needed.

## Why Zip Bomb Detection Happens

Zip bomb detection looks for:
- Very large files (>1GB)
- High compression ratios
- Many nested directories
- Symlinks that create circular references

Our project includes:
- 9.3GB of data files (included in zip)
- 1.5GB virtual environment (excluded)
- 511MB model files (included in zip)

The zip file will be large (10GB+) but should still extract safely. The virtual environment is excluded to reduce size and can be recreated.

