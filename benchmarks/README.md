# Benchmarks

## LoCoMo Evaluation

[LoCoMo](https://arxiv.org/abs/2401.14325) tests multi-hop reasoning over long conversation histories.

### Dataset

Download the LoCoMo dataset and format as JSON:

```json
[
  {
    "conversation": [
      {"role": "user", "content": "I just got a new puppy named Max."},
      {"role": "assistant", "content": "Congratulations! What breed is Max?"},
      {"role": "user", "content": "He's a golden retriever."}
    ],
    "questions": [
      {"question": "What is the user's pet's name?", "answer": "Max"},
      {"question": "What breed is the pet?", "answer": "golden retriever"}
    ]
  }
]
```

### Running

```bash
python -m benchmarks.locomo_eval --data path/to/locomo.json --top-k 20
python -m benchmarks.locomo_eval --data path/to/locomo.json --top-k 20 --output results.json
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | % of questions where expected answer found in recalled memories | ≥ 29.0% |
| Avg Recall (ms) | Average time to recall top-K memories per query | < 100ms |
