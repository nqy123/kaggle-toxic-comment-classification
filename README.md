# Kaggle Toxic Comment Classification Challenge

This repository contains a legal, reproducible solution for Kaggle's
`jigsaw-toxic-comment-classification-challenge`.

The current best version combines a strong TF-IDF NB-SVM baseline with GPU
deep-learning text models. No post-competition `test_labels.csv` leakage is used.

## Best Result

| File | Public AUC | Private AUC |
| --- | ---: | ---: |
| `outputs/submission_nbsvm_keras20_transformer20.csv` | 0.98226 | 0.98240 |

The downloaded public leaderboard snapshot puts the top 20% threshold around
`0.9867`. This version improves clearly over the TF-IDF baseline, but it is still
below the requested top 20% target.

## Method

The final selected submission is a probability blend:

- 60% TF-IDF NB-SVM Logistic Regression.
- 20% Keras BiGRU-CNN trained on Kaggle GPU.
- 20% DistilBERT Transformer trained on Kaggle GPU.

Feature and modeling details:

- Word TF-IDF with 1-2 grams.
- Character TF-IDF with 3-5 grams.
- Per-label NB-SVM log-count ratio features.
- Per-label Logistic Regression with 5-fold validation.
- Keras `TextVectorization` model with BiGRU and convolution pooling.
- DistilBERT multi-label classifier with sigmoid outputs.
- OOF-based blending for the NB-SVM and Keras components.
- Public/private leaderboard checks only for final submission selection.

## Scores

| Experiment | Submission | Public AUC | Private AUC | Notes |
| --- | --- | ---: | ---: | --- |
| NB-SVM TF-IDF | `outputs/submission_nbsvm_tfidf.csv` | 0.97794 | 0.97937 | Strong legal sparse-text baseline |
| NB-SVM + Keras | `outputs/submission_nbsvm_keras_oof_blend.csv` | 0.97984 | 0.98047 | OOF blend, Keras weight 20% |
| NB-SVM + Keras + Transformer | `outputs/submission_nbsvm_keras20_transformer20.csv` | 0.98226 | 0.98240 | Current best private score |

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
- `outputs/submission_nbsvm_keras_oof_blend.csv`
- `outputs/submission_nbsvm_tfidf.csv`
- `outputs/oof_nbsvm_tfidf.csv`
- `outputs/pred_nbsvm_tfidf.csv`
- `outputs/kernel_keras_gpu/`
- `outputs/kernel_transformer_gpu/`
- `outputs/experiment_log.csv`
- `outputs/best_result_summary.csv`

Raw Kaggle data is not committed.
