import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Kaggle may assign P100. Install a PyTorch build that supports sm_60.
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] installing CUDA-compatible torch wheel", flush=True)
install_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-cache-dir",
        "--force-reinstall",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ],
    check=False,
)
print(
    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] torch install return code={install_result.returncode}",
    flush=True,
)

import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 3


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_csv(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if not matches:
        matches = list(Path("/kaggle/input").rglob(f"{name}.zip"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name}")
    return matches[0]


def clean_text(text):
    text = str(text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = text.replace("&amp;", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CommentDataset(Dataset):
    def __init__(self, texts, tokenizer, labels=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(preds)


def main():
    seed_everything()
    train = pd.read_csv(find_csv("train.csv"))
    test = pd.read_csv(find_csv("test.csv"))
    sample = pd.read_csv(find_csv("sample_submission.csv"))
    train["comment_text"] = train["comment_text"].fillna("").map(clean_text)
    test["comment_text"] = test["comment_text"].fillna("").map(clean_text)
    y = train[TARGETS].values.astype("float32")
    stratify_y = train[TARGETS].sum(axis=1).clip(0, 1).values

    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=0.1, random_state=SEED, stratify=stratify_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(TARGETS),
        problem_type="multi_label_classification",
    ).to(device)

    train_loader = DataLoader(CommentDataset(train["comment_text"].values[tr_idx], tokenizer, y[tr_idx]), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(CommentDataset(train["comment_text"].values[va_idx], tokenizer, y[va_idx]), batch_size=BATCH_SIZE * 2, shuffle=False)
    test_loader = DataLoader(CommentDataset(test["comment_text"].values, tokenizer), batch_size=BATCH_SIZE * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.06), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = -1
    best_test = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 200 == 0 or step == len(train_loader):
                log(f"epoch {epoch}: step {step}/{len(train_loader)}, loss={np.mean(losses[-200:]):.6f}")
        val_pred = predict(model, valid_loader, device)
        val_auc = roc_auc_score(y[va_idx], val_pred, average="macro")
        log(f"epoch {epoch}: loss={np.mean(losses):.6f}, val_macro_auc={val_auc:.6f}")
        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_test = predict(model, test_loader, device)

    sub = sample.copy()
    sub[TARGETS] = np.clip(best_test, 0, 1)
    sub.to_csv("submission_transformer_holdout_e3.csv", index=False)
    pd.DataFrame(best_test, columns=TARGETS).assign(id=test["id"]).to_csv("pred_transformer_holdout_e3.csv", index=False)
    pd.DataFrame({"best_val_auc": [best_auc], "epochs": [EPOCHS], "max_len": [MAX_LEN]}).to_csv("transformer_holdout_e3_summary.csv", index=False)
    log(f"saved submission_transformer_holdout_e3.csv best_val_auc={best_auc:.6f}")


if __name__ == "__main__":
    main()
