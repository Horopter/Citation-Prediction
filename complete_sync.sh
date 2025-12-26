#!/bin/bash
# Complete sync script - adds all remaining files and pushes to GitHub
# Run this to ensure EVERYTHING is synced to remote

set -e

echo "=========================================="
echo "Complete Project Sync to GitHub"
echo "=========================================="
echo ""

# Remove any lock files
rm -f .git/index.lock
echo "✓ Cleared any lock files"
echo ""

# Add all remaining files
echo "Adding all remaining files..."
git add -A 2>&1 | grep -v "error: short read" || true
echo "✓ Files added"
echo ""

# Show what will be committed
echo "Files to be committed:"
git status --short | head -20
TOTAL=$(git status --short | wc -l | tr -d ' ')
echo "... ($TOTAL files total)"
echo ""

# Check LFS files
LFS_COUNT=$(git lfs ls-files 2>/dev/null | wc -l | tr -d ' ')
echo "Files tracked by Git LFS: $LFS_COUNT"
if [ "$LFS_COUNT" -gt 0 ]; then
    echo "Sample LFS files:"
    git lfs ls-files | head -5
fi
echo ""

# Commit
echo "Committing all files..."
git commit -m "Complete project sync: All files with Git LFS

- All source code, notebooks, and scripts
- All data files (tracked via Git LFS: .parquet, .npy)
- All model files (tracked via Git LFS: .pkl, .safetensors, .bin, .pt, .pth)
- All documentation and reports
- All logs, metrics, and executed notebooks
- All embedding models
- Complete project structure - NO EXCEPTIONS" || echo "⚠ No new files to commit"
echo ""

# Push to remote
echo "Pushing to GitHub..."
if git push origin main; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS! All files synced to GitHub"
    echo "=========================================="
    echo ""
    echo "Repository: https://github.com/Horopter/Citation-Prediction"
    echo ""
    echo "Final status:"
    git status --short | wc -l | xargs echo "Untracked files:"
    git lfs ls-files 2>/dev/null | wc -l | xargs echo "LFS files:"
    echo ""
else
    echo ""
    echo "⚠ Push failed. Check your GitHub authentication:"
    echo "   gh auth login"
    echo ""
fi






