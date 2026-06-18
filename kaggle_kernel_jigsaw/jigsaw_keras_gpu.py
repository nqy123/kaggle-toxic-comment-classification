import gc
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_TOKENS = 220000
SEQ_LEN = 220
BATCH_SIZE = 256
EPOCHS = 3
N_SPLITS = 3


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def find_csv(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if not matches:
        matches = list(Path("/kaggle/input").rglob(f"{name}.zip"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name}. Existing roots={list(Path('/kaggle/input').glob('*'))}")
    return matches[0]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"[^a-z0-9!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_model(vectorizer):
    text_in = tf.keras.Input(shape=(1,), dtype=tf.string, name="comment_text")
    x = vectorizer(text_in)
    x = tf.keras.layers.Embedding(MAX_TOKENS, 160, mask_zero=False)(x)
    x = tf.keras.layers.SpatialDropout1D(0.25)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(80, return_sequences=True))(x)
    c3 = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    c5 = tf.keras.layers.Conv1D(96, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.Concatenate()([x, c3, c5])
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    x = tf.keras.layers.Dense(192, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(len(TARGETS), activation="sigmoid")(x)
    model = tf.keras.Model(text_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def main():
    seed_everything()
    log(f"tensorflow={tf.__version__}")
    log(f"gpus={tf.config.list_physical_devices('GPU')}")

    train = pd.read_csv(find_csv("train.csv"))
    test = pd.read_csv(find_csv("test.csv"))
    sample = pd.read_csv(find_csv("sample_submission.csv"))
    train["comment_text"] = train["comment_text"].fillna("").map(clean_text)
    test["comment_text"] = test["comment_text"].fillna("").map(clean_text)
    y = train[TARGETS].values.astype("float32")

    log("adapt TextVectorization")
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQ_LEN,
        standardize=None,
        split="whitespace",
    )
    text_ds = tf.data.Dataset.from_tensor_slices(train["comment_text"].values).batch(1024)
    vectorizer.adapt(text_ds)

    oof = np.zeros((len(train), len(TARGETS)), dtype="float32")
    pred = np.zeros((len(test), len(TARGETS)), dtype="float32")
    fold_scores = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        log(f"========== fold {fold}/{N_SPLITS} ==========")
        model = build_model(vectorizer)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=1, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=1, min_lr=2e-5),
        ]
        model.fit(
            train["comment_text"].values[tr_idx],
            y[tr_idx],
            validation_data=(train["comment_text"].values[va_idx], y[va_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )
        va_pred = model.predict(train["comment_text"].values[va_idx], batch_size=BATCH_SIZE, verbose=0)
        te_pred = model.predict(test["comment_text"].values, batch_size=BATCH_SIZE, verbose=0)
        fold_auc = roc_auc_score(y[va_idx], va_pred, average="macro")
        log(f"fold {fold} macro AUC={fold_auc:.6f}")
        oof[va_idx] = va_pred
        pred += te_pred / N_SPLITS
        fold_scores.append(float(fold_auc))
        tf.keras.backend.clear_session()
        gc.collect()

    oof_auc = roc_auc_score(y, oof, average="macro")
    log(f"OOF macro AUC={oof_auc:.6f}, folds={fold_scores}")
    oof_df = pd.DataFrame(oof, columns=TARGETS)
    oof_df.insert(0, "id", train["id"])
    pred_df = pd.DataFrame(pred, columns=TARGETS)
    pred_df.insert(0, "id", test["id"])
    sub = sample.copy()
    sub[TARGETS] = np.clip(pred, 0, 1)
    oof_df.to_csv("oof_keras_bigru_cnn.csv", index=False)
    pred_df.to_csv("pred_keras_bigru_cnn.csv", index=False)
    sub.to_csv("submission_keras_bigru_cnn.csv", index=False)
    pd.DataFrame({"fold": range(1, N_SPLITS + 1), "auc": fold_scores}).to_csv("keras_bigru_cnn_fold_scores.csv", index=False)


if __name__ == "__main__":
    main()
