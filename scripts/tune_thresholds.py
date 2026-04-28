# scripts/tune_thresholds.py — 修复版
import os
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# ★ 自动切换到项目根目录（E:\ai）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)12
print(f"工作目录: {os.getcwd()}")
LABELS = ["ad", "abuse", "negative", "misinfo"]

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def batched_predict_logits(model, tokenizer, texts, device, max_length=256, batch_size=32):
    all_logits = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)
    return np.vstack(all_logits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/campus_detector")
    parser.add_argument("--val_file", type=str, default="data/val.jsonl")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_file", type=str, default="models/campus_detector/thresholds.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    samples = read_jsonl(args.val_file)
    texts = [str(x[args.text_key]) for x in samples]
    y_true = np.array([[int(x[l]) for l in LABELS] for x in samples], dtype=int)

    logits = batched_predict_logits(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    probs = sigmoid(logits)

    # 每类独立搜阈值，目标：该类F1最大
    thresholds = {}
    per_label_metrics = {}
    grid = np.arange(0.05, 0.96, 0.01)

    for i, lab in enumerate(LABELS):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred_i = (probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred_i, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))

        thresholds[lab] = best_t
        pred_best = (probs[:, i] >= best_t).astype(int)
        per_label_metrics[lab] = {
            "f1": float(f1_score(y_true[:, i], pred_best, zero_division=0)),
            "precision": float(precision_score(y_true[:, i], pred_best, zero_division=0)),
            "recall": float(recall_score(y_true[:, i], pred_best, zero_division=0)),
        }

    # 计算整体指标
    pred_all = np.zeros_like(y_true)
    for i, lab in enumerate(LABELS):
        pred_all[:, i] = (probs[:, i] >= thresholds[lab]).astype(int)

    macro_f1 = float(f1_score(y_true, pred_all, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, pred_all, average="micro", zero_division=0))

    result = {
        "labels": LABELS,
        "thresholds": thresholds,
        "metrics": {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "per_label": per_label_metrics
        }
    }

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    with open(args.save_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== Threshold tuning finished ===")
    print("Best thresholds:", thresholds)
    print("macro_f1:", macro_f1, "micro_f1:", micro_f1)
    print(f"Saved to: {args.save_file}")

if __name__ == "__main__":
    main()