# scripts/build_dataset_v2.py — 修复版
import os
import re
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

RAW_DIR = "data/raw"
OUT_TRAIN = "data/train.jsonl"
OUT_VAL = "data/val.jsonl"

# ★ 修复：统一为 negative，与 train.py / api.py 一致
LABELS = ["ad", "abuse", "negative", "misinfo"]


def read_csv_auto(path):
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Failed to read csv: {path}")


def norm_text(x):
    x = str(x) if x is not None else ""
    x = re.sub(r"\s+", " ", x).strip()
    return x


def add_sample(samples, text, ad=0, abuse=0, negative=0, misinfo=0, source="unknown"):
    """★ 修复：politic 参数改为 negative"""
    text = norm_text(text)
    if not text or len(text) < 4:  # 过滤过短文本
        return
    samples.append({
        "text": text,
        "ad": int(ad),
        "abuse": int(abuse),
        "negative": int(negative),  # ★ 修复
        "misinfo": int(misinfo),
        "_source": source
    })


# ==================== 中文数据加载器（新增） ====================

def load_chinese_custom(samples):
    """
    加载你自己爬取的中文数据。
    期望格式：data/raw/chinese_campus.jsonl
    每行格式：{"text": "...", "ad": 0, "abuse": 1, "negative": 0, "misinfo": 0}
    """
    path = os.path.join(RAW_DIR, "chinese_campus.jsonl")
    if not os.path.exists(path):
        print("[skip] chinese_campus.jsonl not found")
        return
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            add_sample(
                samples,
                obj.get("text", ""),
                ad=obj.get("ad", 0),
                abuse=obj.get("abuse", 0),
                negative=obj.get("negative", 0),
                misinfo=obj.get("misinfo", 0),
                source="chinese_campus"
            )
            count += 1
    print(f"[loaded] chinese_campus.jsonl: {count} samples")


def load_cold_offensive(samples):
    """
    加载 COLD (Chinese Offensive Language Dataset)
    GitHub: https://github.com/thu-coai/COLDataset
    期望格式：data/raw/cold.jsonl 或 cold.csv
    """
    jsonl_path = os.path.join(RAW_DIR, "cold.jsonl")
    csv_path = os.path.join(RAW_DIR, "cold.csv")

    if os.path.exists(jsonl_path):
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", obj.get("TEXT", ""))
                label = int(obj.get("label", obj.get("Label", 0)))
                if label == 1:
                    add_sample(samples, text, abuse=1, source="cold")
                else:
                    add_sample(samples, text, source="cold")
                count += 1
        print(f"[loaded] cold.jsonl: {count} samples")
    elif os.path.exists(csv_path):
        df = read_csv_auto(csv_path)
        for _, r in df.iterrows():
            text = str(r.get("TEXT", r.get("text", "")))
            label = int(r.get("Label", r.get("label", 0)))
            if label == 1:
                add_sample(samples, text, abuse=1, source="cold")
            else:
                add_sample(samples, text, source="cold")
        print(f"[loaded] cold.csv: {len(df)} samples")
    else:
        print("[skip] cold dataset not found")


def load_weibo_rumor(samples):
    """
    加载微博谣言数据集
    期望格式：data/raw/weibo_rumor.jsonl
    每行：{"text": "...", "label": 1}  (1=谣言, 0=非谣言)
    """
    path = os.path.join(RAW_DIR, "weibo_rumor.jsonl")
    if not os.path.exists(path):
        print("[skip] weibo_rumor.jsonl not found")
        return
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            label = int(obj.get("label", 0))
            if label == 1:
                add_sample(samples, text, misinfo=1, source="weibo_rumor")
            else:
                add_sample(samples, text, source="weibo_rumor")
            count += 1
    print(f"[loaded] weibo_rumor.jsonl: {count} samples")


def load_chinese_spam(samples):
    """
    加载中文垃圾短信/广告数据集
    期望格式：data/raw/chinese_spam.csv
    列：text, label (1=spam/ad, 0=ham)
    """
    path = os.path.join(RAW_DIR, "chinese_spam.csv")
    if not os.path.exists(path):
        print("[skip] chinese_spam.csv not found")
        return
    df = read_csv_auto(path)
    for _, r in df.iterrows():
        text = str(r.get("text", r.get("sms", "")))
        label = int(r.get("label", 0))
        if label == 1:
            add_sample(samples, text, ad=1, source="chinese_spam")
        else:
            add_sample(samples, text, source="chinese_spam")
    print(f"[loaded] chinese_spam.csv: {len(df)} samples")


# ==================== 英文数据加载器（保留原有） ====================

def load_sms_spam(samples):
    path = os.path.join(RAW_DIR, "sms_spam.csv")
    if not os.path.exists(path):
        print("[skip] sms_spam.csv not found")
        return
    df = read_csv_auto(path)
    cols = {c.lower(): c for c in df.columns}
    if "v1" in cols and "v2" in cols:
        label_col, text_col = cols["v1"], cols["v2"]
    elif "label" in cols and "text" in cols:
        label_col, text_col = cols["label"], cols["text"]
    else:
        print("[skip] sms_spam.csv columns not recognized")
        return
    for _, r in df.iterrows():
        lbl = str(r[label_col]).lower()
        txt = r[text_col]
        if "spam" in lbl:
            add_sample(samples, txt, ad=1, source="sms_spam")
        else:
            add_sample(samples, txt, source="sms_spam")


def load_youtube_spam(samples):
    folder = os.path.join(RAW_DIR, "youtube_spam")
    if not os.path.exists(folder):
        print("[skip] youtube_spam folder not found")
        return
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not files:
        print("[skip] no csv in youtube_spam folder")
        return
    for fn in files:
        path = os.path.join(folder, fn)
        df = read_csv_auto(path)
        cols = {c.lower(): c for c in df.columns}
        content_col = cols.get("content")
        class_col = cols.get("class")
        if content_col is None or class_col is None:
            continue
        for _, r in df.iterrows():
            txt = r[content_col]
            cls = int(r[class_col])
            if cls == 1:
                add_sample(samples, txt, ad=1, source="youtube_spam")
            else:
                add_sample(samples, txt, source="youtube_spam")


def load_jigsaw(samples):
    path = os.path.join(RAW_DIR, "jigsaw_train.csv")
    if not os.path.exists(path):
        print("[skip] jigsaw_train.csv not found")
        return
    df = read_csv_auto(path)
    required = ["comment_text", "toxic", "severe_toxic", "obscene",
                 "threat", "insult", "identity_hate"]
    if any(c not in df.columns for c in required):
        print("[skip] jigsaw_train.csv columns not recognized")
        return
    for _, r in df.iterrows():
        txt = r["comment_text"]
        toxic_sum = sum(float(r[c]) for c in required[1:])
        if toxic_sum > 0:
            add_sample(samples, txt, abuse=1, source="jigsaw")
        else:
            add_sample(samples, txt, source="jigsaw")


def load_ag_news(samples):
    path = os.path.join(RAW_DIR, "ag_news_train.csv")
    if not os.path.exists(path):
        print("[skip] ag_news_train.csv not found")
        return
    df = read_csv_auto(path)
    cols = {c.lower(): c for c in df.columns}
    class_col = cols.get("class index") or cols.get("class_index") or cols.get("label")
    title_col = cols.get("title")
    desc_col = cols.get("description")
    if class_col is None:
        if df.shape[1] >= 3:
            df.columns = (["Class Index", "Title", "Description"]
                          + [f"extra_{i}" for i in range(df.shape[1] - 3)])
            class_col, title_col, desc_col = "Class Index", "Title", "Description"
        else:
            print("[skip] ag_news_train.csv columns not recognized")
            return
    for _, r in df.iterrows():
        cls = int(r[class_col])
        title = str(r[title_col]) if title_col else ""
        desc = str(r[desc_col]) if desc_col else ""
        txt = (title + " " + desc).strip()
        # ★ 修复：原来映射为 politic，现改为 negative
        if cls == 1:
            add_sample(samples, txt, negative=1, source="ag_news")
        else:
            add_sample(samples, txt, source="ag_news")


def load_liar(samples):
    path = os.path.join(RAW_DIR, "liar_train.tsv")
    if not os.path.exists(path):
        print("[skip] liar_train.tsv not found")
        return
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 3:
        print("[skip] liar_train.tsv format not recognized")
        return
    fake_labels = {"pants-fire", "false", "barely-true"}
    for _, r in df.iterrows():
        lbl = str(r[1]).strip().lower()
        txt = r[2]
        if lbl in fake_labels:
            add_sample(samples, txt, misinfo=1, source="liar")
        else:
            add_sample(samples, txt, source="liar")


def load_fake_real_news(samples):
    fake_path = os.path.join(RAW_DIR, "fake_news_fake.csv")
    true_path = os.path.join(RAW_DIR, "fake_news_true.csv")
    if os.path.exists(fake_path):
        df = read_csv_auto(fake_path)
        for _, r in df.iterrows():
            title = str(r["title"]) if "title" in df.columns else ""
            text = str(r["text"]) if "text" in df.columns else ""
            add_sample(samples, (title + " " + text).strip(),
                       misinfo=1, source="fake_news_fake")
    else:
        print("[skip] fake_news_fake.csv not found")
    if os.path.exists(true_path):
        df = read_csv_auto(true_path)
        for _, r in df.iterrows():
            title = str(r["title"]) if "title" in df.columns else ""
            text = str(r["text"]) if "text" in df.columns else ""
            add_sample(samples, (title + " " + text).strip(),
                       source="fake_news_true")
    else:
        print("[skip] fake_news_true.csv not found")


# ==================== 工具函数 ====================

def dedup(samples):
    seen = set()
    out = []
    for x in samples:
        key = x["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def label_stats(data):
    return {
        "n": len(data),
        "ad": sum(x["ad"] for x in data),
        "abuse": sum(x["abuse"] for x in data),
        "negative": sum(x["negative"] for x in data),  # ★ 修复
        "misinfo": sum(x["misinfo"] for x in data),
    }


def strat_key(x):
    return f'{x["ad"]}{x["abuse"]}{x["negative"]}{x["misinfo"]}'  # ★ 修复


def balance_dataset(samples, neg_ratio=1.2):
    pos_idx = set()
    for i, x in enumerate(samples):
        if x["ad"] or x["abuse"] or x["negative"] or x["misinfo"]:  # ★ 修复
            pos_idx.add(i)
    pos_samples = [samples[i] for i in sorted(pos_idx)]
    neg_samples = [samples[i] for i in range(len(samples)) if i not in pos_idx]
    target_neg = int(len(pos_samples) * neg_ratio)
    if len(neg_samples) > target_neg:
        neg_samples = random.sample(neg_samples, target_neg)
    out = pos_samples + neg_samples
    random.shuffle(out)
    return out


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            obj = {
                "text": x["text"],
                "ad": x["ad"],
                "abuse": x["abuse"],
                "negative": x["negative"],  # ★ 修复
                "misinfo": x["misinfo"]
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    samples = []

    # ★ 优先加载中文数据
    print("=== 加载中文数据 ===")
    load_chinese_custom(samples)
    load_cold_offensive(samples)
    load_weibo_rumor(samples)
    load_chinese_spam(samples)

    # 英文数据（补充用，但效果不如中文数据）
    print("\n=== 加载英文补充数据 ===")
    load_sms_spam(samples)
    load_youtube_spam(samples)
    load_jigsaw(samples)
    load_ag_news(samples)
    load_liar(samples)
    load_fake_real_news(samples)

    samples = dedup(samples)
    print("\n[after dedup]", label_stats(samples))

    samples = balance_dataset(samples, neg_ratio=1.2)
    print("[after balance]", label_stats(samples))

    keys = [strat_key(x) for x in samples]
    use_stratify = True
    vc = pd.Series(keys).value_counts()
    if (vc < 2).any():
        use_stratify = False

    if use_stratify:
        train_data, val_data = train_test_split(
            samples, test_size=0.14, random_state=42, stratify=keys
        )
    else:
        train_data, val_data = train_test_split(
            samples, test_size=0.14, random_state=42
        )

    save_jsonl(train_data, OUT_TRAIN)
    save_jsonl(val_data, OUT_VAL)

    print("\n[train]", label_stats(train_data))
    print("[val]", label_stats(val_data))
    print(f"Saved: {OUT_TRAIN}")
    print(f"Saved: {OUT_VAL}")


if __name__ == "__main__":
    main()
