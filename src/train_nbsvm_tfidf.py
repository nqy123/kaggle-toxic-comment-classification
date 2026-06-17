import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "jigsaw-toxic-comment-classification-challenge"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ID_COL = "id"
TEXT_COL = "comment_text"
TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
SEED = 42


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def find_csv(name):
    """兼容 Kaggle 解压后出现 csv 同名目录的情况。"""
    direct = DATA_DIR / name
    nested = DATA_DIR / name / name
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"cannot find {name}")


def clean_text(s):
    """轻量文本清洗，保留脏词、标点和大小写信息给 char n-gram。"""
    s = str(s).lower()
    s = re.sub(r"\n", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def nb_log_count_ratio(x, y):
    """NB-SVM 的 log-count ratio，对每个标签单独计算。"""
    p = x[y == 1].sum(axis=0) + 1.0
    q = x[y == 0].sum(axis=0) + 1.0
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    return np.log((p / p.sum()) / (q / q.sum()))


def fit_one_label(x_train, x_test, y, label):
    """每个标签做 5 折 OOF，并用全量训练集拟合最终模型。"""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(x_train.shape[0], dtype=np.float32)
    test_pred = np.zeros(x_test.shape[0], dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(x_train, y), 1):
        log(f"{label} fold {fold}/5")
        r = nb_log_count_ratio(x_train[tr_idx], y[tr_idx])
        x_tr = x_train[tr_idx].multiply(r)
        x_va = x_train[va_idx].multiply(r)
        x_te = x_test.multiply(r)
        model = LogisticRegression(C=4.0, solver="liblinear", max_iter=250, random_state=SEED)
        model.fit(x_tr, y[tr_idx])
        va_pred = model.predict_proba(x_va)[:, 1]
        te_pred = model.predict_proba(x_te)[:, 1]
        auc = roc_auc_score(y[va_idx], va_pred)
        log(f"{label} fold {fold} AUC={auc:.6f}")
        oof[va_idx] = va_pred
        test_pred += te_pred / folds.n_splits
        fold_scores.append(float(auc))

    oof_auc = roc_auc_score(y, oof)
    log(f"{label} OOF AUC={oof_auc:.6f}")
    return oof, test_pred, oof_auc, fold_scores


def main():
    total_start = time.perf_counter()
    log("read data")
    train = pd.read_csv(find_csv("train.csv"))
    test = pd.read_csv(find_csv("test.csv"))
    sample = pd.read_csv(find_csv("sample_submission.csv"))

    train_text = train[TEXT_COL].fillna("").map(clean_text)
    test_text = test[TEXT_COL].fillna("").map(clean_text)
    all_text = pd.concat([train_text, test_text], axis=0, ignore_index=True)

    log("fit word tfidf")
    word_vec = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        ngram_range=(1, 2),
        max_features=220000,
        min_df=2,
        sublinear_tf=True,
    )
    word_all = word_vec.fit_transform(all_text)

    log("fit char tfidf")
    char_vec = TfidfVectorizer(
        strip_accents="unicode",
        analyzer="char",
        ngram_range=(3, 5),
        max_features=320000,
        min_df=2,
        sublinear_tf=True,
    )
    char_all = char_vec.fit_transform(all_text)

    log("stack sparse features")
    x_all = sparse.hstack([word_all, char_all], format="csr")
    x_train = x_all[: len(train)]
    x_test = x_all[len(train):]
    log(f"x_train={x_train.shape}, x_test={x_test.shape}, nnz={x_train.nnz + x_test.nnz}")

    oof_df = pd.DataFrame({ID_COL: train[ID_COL]})
    pred_df = pd.DataFrame({ID_COL: test[ID_COL]})
    sub = sample.copy()
    scores = {}

    for label in TARGETS:
        y = train[label].values.astype(int)
        oof, pred, oof_auc, fold_scores = fit_one_label(x_train, x_test, y, label)
        oof_df[label] = oof
        pred_df[label] = pred
        sub[label] = np.clip(pred, 0, 1)
        scores[label] = {"oof_auc": float(oof_auc), "fold_scores": fold_scores}

    mean_auc = float(np.mean([scores[label]["oof_auc"] for label in TARGETS]))
    log(f"mean OOF AUC={mean_auc:.6f}")

    oof_path = OUTPUT_DIR / "oof_nbsvm_tfidf.csv"
    pred_path = OUTPUT_DIR / "pred_nbsvm_tfidf.csv"
    sub_path = OUTPUT_DIR / "submission_nbsvm_tfidf.csv"
    oof_df.to_csv(oof_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    sub.to_csv(sub_path, index=False)

    summary = {
        "model": "NB-SVM LogisticRegression",
        "features": "word 1-2 tfidf + char 3-5 tfidf",
        "mean_oof_auc": mean_auc,
        "label_scores": scores,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(x_train.shape[1]),
        "notes": "Sparse TF-IDF linear model; GPU is not used because this high-scoring baseline is sparse linear rather than tree/deep learning.",
        "elapsed_seconds": time.perf_counter() - total_start,
    }
    (OUTPUT_DIR / "nbsvm_tfidf_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {sub_path}")
    log(f"TOTAL elapsed={time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
