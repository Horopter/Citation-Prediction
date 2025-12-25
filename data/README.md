# Kaggle Dataset: f-25-si-670-kaggle-2

## Dataset Overview

This dataset contains academic papers from OpenAlex in JSONL format. The task appears to be a binary classification problem.

## Files

- **train.jsonl**: 960,000 training samples with labels
- **val.jsonl**: 120,000 validation samples with labels  
- **test.no_label.jsonl**: 120,000 test samples without labels

## Data Structure

Each record contains:
- **Metadata**: `work_id`, `id`, `title`, `display_name`, `type`, `publication_year`, `publication_date`, etc.
- **Authors**: `authorships` array with author information and affiliations
- **Concepts**: Array of research concepts/topics with relevance scores
- **Abstract**: Text abstract (present in ~55% of records)
- **Open Access Info**: `open_access` object with OA status
- **Locations**: Publication locations and sources
- **Related Works**: Array of related work IDs
- **Label**: Binary label (0 or 1) - only in train and val files

## Label Distribution

From a sample of training data:
- **Label 0**: ~92% (majority class)
- **Label 1**: ~8% (minority class)

This is an imbalanced classification problem.

## Key Statistics (from sample)

- **Publication Years**: 2018-2022
- **Abstract Coverage**: ~55% of records have abstracts
- **Average Concepts**: ~10 concepts per paper
- **Concept Range**: 1-32 concepts per paper

## Data Loading Example

```python
import json

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load training data
train_data = load_jsonl('train.jsonl')
```

