# scripts/predict.py
import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["ad", "abuse", "negative", "misinfo"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_thresholds(model_dir):
    th_path = os.path.join(model_dir, "thresholds.json")
    if not os.path.exists(th_path):
        print("thresholds.json not found, use default 0.5 for all labels.")
        return {l: 0.5 for l in LABELS}
    with open(th_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("thresholds", {l: 0.5 for l in LABELS})

def predict_texts(model, tokenizer, texts, device, max_length=256, batch_size=32):
    model.eval()
    all_probs = []
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
            logits = model(**enc).logits.detach().cpu().numpy()
            probs = sigmoid(logits)
            all_probs.append(probs)
    return np.vstack(all_probs)

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/campus_detector")
    parser.add_argument("--text", type=str, default=None, help="single text for prediction")
    parser.add_argument("--input_file", type=str, default=None, help="jsonl file for batch prediction")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--output_file", type=str, default="predictions.jsonl")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.text is None and args.input_file is None:
        raise ValueError("Please provide --text or --input_file")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ★ 自动切到项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    print(f"工作目录: {os.getcwd()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(device)

    thresholds = load_thresholds(args.model_dir)
    print("Thresholds:", thresholds)

    # 单条预测
    if args.text is not None:
        probs = predict_texts(
            model, tokenizer, [args.text], device,
            max_length=args.max_length, batch_size=1
        )[0]

        prob_dict = {lab: float(probs[i]) for i, lab in enumerate(LABELS)}
        pred_bin = {lab: int(probs[i] >= thresholds.get(lab, 0.5)) for i, lab in enumerate(LABELS)}
        pred_labels = [lab for lab in LABELS if pred_bin[lab] == 1]

        print("\n=== Prediction ===")
        print("Text:", args.text)
        print("Probabilities:", prob_dict)
        print("Pred Binary:", pred_bin)
        print("Pred Labels:", pred_labels)
        return

    # 批量预测
    samples = read_jsonl(args.input_file)
    texts = [str(x[args.text_key]) for x in samples]
    probs_all = predict_texts(
        model, tokenizer, texts, device,
        max_length=args.max_length, batch_size=args.batch_size
    )

    with open(args.output_file, "w", encoding="utf-8") as f:
        for sample, probs in zip(samples, probs_all):
            prob_dict = {lab: float(probs[i]) for i, lab in enumerate(LABELS)}
            pred_bin = {lab: int(probs[i] >= thresholds.get(lab, 0.5)) for i, lab in enumerate(LABELS)}
            pred_labels = [lab for lab in LABELS if pred_bin[lab] == 1]

            out = {
                "text": sample.get(args.text_key, ""),
                "probabilities": prob_dict,
                "pred_binary": pred_bin,
                "pred_labels": pred_labels
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Batch prediction done. Saved to: {args.output_file}")

if __name__ == "__main__":
    main()