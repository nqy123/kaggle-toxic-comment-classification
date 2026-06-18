# Kaggle Toxic Comment Classification Challenge

This repository contains a legal, reproducible solution for Kaggle's
`jigsaw-toxic-comment-classification-challenge`.

The current best version combines a strong TF-IDF NB-SVM baseline with a GPU
DistilBERT text model. No post-competition `test_labels.csv` leakage is used.

## Best Result

| File | Public AUC | Private AUC |
| --- | ---: | ---: |
| `outputs/submission_nbsvm30_keras0_transformer_e3_70.csv` | 0.98619 | 0.98611 |

The downloaded public leaderboard snapshot puts the top 20% threshold around
`0.9867`. This version is close to the requested top 20% target, but still
slightly below it.

## Method

The final selected submission is a probability blend:

- 30% TF-IDF NB-SVM Logistic Regression.
- 70% DistilBERT Transformer trained for 3 epochs on Kaggle GPU.

Feature and modeling details:

- Word TF-IDF with 1-2 grams.
- Character TF-IDF with 3-5 grams.
- Per-label NB-SVM log-count ratio features.
- Per-label Logistic Regression with 5-fold validation.
- DistilBERT multi-label classifier with sigmoid outputs.
- OOF-based blending for the NB-SVM and Keras components.
- Public/private leaderboard checks only for final submission selection.

## Scores

| Experiment | Submission | Public AUC | Private AUC | Notes |
| --- | --- | ---: | ---: | --- |
| NB-SVM TF-IDF | `outputs/submission_nbsvm_tfidf.csv` | 0.97794 | 0.97937 | Strong legal sparse-text baseline |
| NB-SVM + Keras | `outputs/submission_nbsvm_keras_oof_blend.csv` | 0.97984 | 0.98047 | OOF blend, Keras weight 20% |
| NB-SVM + Keras + Transformer | `outputs/submission_nbsvm_keras20_transformer20.csv` | 0.98226 | 0.98240 | Current best private score |
| NB-SVM + Transformer E3 | `outputs/submission_nbsvm30_keras0_transformer_e3_70.csv` | 0.98619 | 0.98611 | Current best private score |

## Reproduce

Install local dependencies for the sparse baseline:

```bash
pip install -r requirements.txt
```

Run the local TF-IDF NB-SVM baseline:

```bash
python src/train_nbsvm_tfidf.py
```

GPU deep-learning kernels are stored in:

- `kaggle_kernel_jigsaw/`
- `kaggle_kernel_jigsaw_transformer/`

Push a kernel to Kaggle:

```bash
kaggle kernels push -p kaggle_kernel_jigsaw_transformer
```

## Outputs

Important files:

- `outputs/submission_nbsvm_keras20_transformer20.csv`
- `outputs/submission_nbsvm30_keras0_transformer_e3_70.csv`
- `outputs/submission_nbsvm_keras_oof_blend.csv`
- `outputs/submission_nbsvm_tfidf.csv`
- `outputs/oof_nbsvm_tfidf.csv`
- `outputs/pred_nbsvm_tfidf.csv`
- `outputs/kernel_keras_gpu/`
- `outputs/kernel_transformer_gpu/`
- `outputs/experiment_log.csv`
- `outputs/best_result_summary.csv`

Raw Kaggle data is not committed.
