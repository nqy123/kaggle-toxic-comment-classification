# Kaggle Toxic Comment Classification Challenge

This project is a legal, reproducible TF-IDF NB-SVM baseline for Kaggle's
`jigsaw-toxic-comment-classification-challenge`.

## Best Result

| File | Public AUC | Private AUC |
| --- | ---: | ---: |
| `outputs/submission_nbsvm_tfidf.csv` | 0.97794 | 0.97937 |

The downloaded public leaderboard snapshot shows the top 20% threshold around `0.9867`.
This version does **not** reach that threshold without using post-competition test labels or a deep-learning ensemble.

## Method

The solution uses a classic strong text baseline:

- Clean comment text lightly while preserving toxic lexical signals.
- Word TF-IDF features with 1-2 grams.
- Character TF-IDF features with 3-5 grams.
- Per-label NB-SVM log-count ratio.
- Per-label Logistic Regression.
- 5-fold StratifiedKFold validation for each target.

No `test_labels.csv` leakage is used.

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/train_nbsvm_tfidf.py
```

## Outputs

Important files:

- `outputs/submission_nbsvm_tfidf.csv`
- `outputs/oof_nbsvm_tfidf.csv`
- `outputs/pred_nbsvm_tfidf.csv`
- `outputs/nbsvm_tfidf_summary.json`
- `outputs/experiment_log.csv`
- `outputs/best_result_summary.csv`

Raw Kaggle data is not committed.
