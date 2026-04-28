# scripts/rebuild_negative.py
"""
重建 negative 类别数据：
- 保留 ChineseSafe 中 negative=1 的高质量样本
- 用 ChnSentiCorp / weibo_senti_100k 的正面文本替换 negative=0
- 过滤掉所有 NLI/考试题格式

使用前请下载数据集到 data/raw/:
  1. weibo_senti_100k.csv  → https://github.com/SophonPlus/ChineseNlpCorpus
  2. ChnSentiCorp.csv 或从 HuggingFace 下载后转成 csv/jsonl

运行: cd E:\ai && python scripts\rebuild_negative.py
"""
import os
import re
import json
import csv
import random
from sklearn.model_selection import train_test_split

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
print(f"工作目录: {os.getcwd()}")

LABELS = ["ad", "abuse", "negative", "misinfo"]
RAW_DIR = "data/raw"


def norm(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_nlp_task(text):
    """检测 NLI/阅读理解/成语填空等格式化样本"""
    if text is None:
        return False

    # 确保 text 是字符串
    text = str(text)

    patterns = [
        r"是的,不是,或也许", r"是的，不是，或也许",
        r"真的,假的,或未知", r"真的，假的，或未知",
        r"正确,错误,或未知", r"正确，错误，或未知",
        r"必然的,可能的,或不可能", r"必然的，可能的，或不可能",
        r"假设.*我们可以推断", r"假定下面是真的",
        r"给定.*因此", r"给定.*我们应该假定",
        r"仅使用以上描述", r"根据前面的段落",
        r"那么下面的陈述", r"我们这样说有道理吗",
        r"候选成语", r"候选的词语", r"完型填空", r"成语填空",
        r"下划线处", r"哪个成语最符合",
        r"阅读理解", r"阅读文章", r"根据短文内容",
        r"代表了这篇论文的摘要", r"这是正确的吗",
        r"\\n问题[：:]", r"\\n 问题[：:]", r"\\n回答",
        r"来替换这个句子",
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def read_jsonl(path):
    samples = []
    if not os.path.exists(path):
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    s = json.loads(line)
                    # 确保样本有文本字段
                    if "text" in s and s["text"] is not None:
                        samples.append(s)
                    else:
                        print(f"警告: 第{i}行缺少文本字段")
                except json.JSONDecodeError as e:
                    print(f"警告: 第{i}行JSON解析错误: {e}")
    return samples


def save_jsonl(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            obj = {
                "text": s["text"],
                "ad": int(s.get("ad", 0)),
                "abuse": int(s.get("abuse", 0)),
                "negative": int(s.get("negative", 0)),
                "misinfo": int(s.get("misinfo", 0)),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ==================== 加载正面/安全文本数据集 ====================

def load_weibo_positive(max_count=None):
    """加载微博正面情感文本（label=1）"""
    path = os.path.join(RAW_DIR, "weibo_senti_100k.csv")
    if not os.path.exists(path):
        print("[skip] weibo_senti_100k.csv 未找到")
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row.get("label", 0))
            text = norm(row.get("review", row.get("text", "")))
            if label == 1 and len(text) >= 6:
                samples.append(text)

    random.shuffle(samples)
    if max_count and len(samples) > max_count:
        samples = samples[:max_count]
    print(f"[loaded] 微博正面文本: {len(samples)} 条")
    return samples


def load_chnsenticorp_positive(max_count=None):
    """
    加载 ChnSentiCorp 正面评论
    支持多种格式：
    - csv: label, text 列
    - tsv: label\ttext
    - jsonl: {"label": 1, "text": "..."}
    """
    possible_paths = [
        os.path.join(RAW_DIR, "ChnSentiCorp.csv"),
        os.path.join(RAW_DIR, "chnsenticorp.csv"),
        os.path.join(RAW_DIR, "ChnSentiCorp_htl_all.csv"),
        os.path.join(RAW_DIR, "ChnSentiCorp.jsonl"),
        os.path.join(RAW_DIR, "ChnSentiCorp.tsv"),
    ]

    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        print("[skip] ChnSentiCorp 未找到")
        return []

    samples = []

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label = int(obj.get("label", 0))
                text = norm(obj.get("text", obj.get("sentence", "")))
                if label == 1 and len(text) >= 6:
                    samples.append(text)

    elif path.endswith(".tsv"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        label = int(parts[0])
                        text = norm(parts[1])
                        if label == 1 and len(text) >= 6:
                            samples.append(text)
                    except ValueError:
                        continue

    else:  # csv
        for enc in ["utf-8", "utf-8-sig", "gbk", "latin-1"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = int(row.get("label", 0))
                        text = norm(row.get("text", row.get("review",
                                                            row.get("sentence", ""))))
                        if label == 1 and len(text) >= 6:
                            samples.append(text)
                break
            except Exception:
                continue

    random.shuffle(samples)
    if max_count and len(samples) > max_count:
        samples = samples[:max_count]
    print(f"[loaded] ChnSentiCorp 正面文本: {len(samples)} 条")
    return samples


def load_toutiao_titles(max_count=None):
    """
    加载今日头条新闻标题（全是正常文本）
    格式：id_类别id_类别名_关键词\ttitle
    下载: https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset
    放到: data/raw/toutiao_cat_data.txt
    """
    path = os.path.join(RAW_DIR, "toutiao_cat_data.txt")
    if not os.path.exists(path):
        print("[skip] toutiao_cat_data.txt 未找到")
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("_!_")
            if len(parts) >= 4:
                text = norm(parts[3])
                if len(text) >= 6:
                    samples.append(text)

    random.shuffle(samples)
    if max_count and len(samples) > max_count:
        samples = samples[:max_count]
    print(f"[loaded] 今日头条标题: {len(samples)} 条")
    return samples


# ==================== 主流程 ====================

def main():
    # ===== 1. 加载现有训练+验证数据 =====
    print("=" * 50)
    print("步骤1：加载现有数据")
    print("=" * 50)

    train_data = read_jsonl("data/train.jsonl")
    val_data = read_jsonl("data/val.jsonl")
    all_data = train_data + val_data
    print(f"现有数据总量: {len(all_data)}")

    # 过滤掉没有文本的样本
    all_data = [s for s in all_data if s.get("text") is not None]
    print(f"过滤后数据量（有文本的）: {len(all_data)}")

    # ===== 2. 分离数据 =====
    print("\n" + "=" * 50)
    print("步骤2：分离数据")
    print("=" * 50)

    # 保留的：其他类别数据（ad/abuse/misinfo 相关的，不动）
    other_samples = []
    # 保留的：negative=1 且不是 NLI 格式的
    neg1_keep = []
    # 丢弃的：negative=0 的 NLI 格式 + 其他要替换的
    neg0_discard = 0
    neg1_discard_nlp = 0

    for s in all_data:
        # 确保文本是字符串
        s["text"] = str(s.get("text", ""))
        # 规范化文本
        s["text"] = norm(s["text"])

        has_other_label = (s.get("ad", 0) == 1 or
                           s.get("abuse", 0) == 1 or
                           s.get("misinfo", 0) == 1)

        if s.get("negative", 0) == 1:
            # negative=1 的样本
            if is_nlp_task(s["text"]):
                neg1_discard_nlp += 1  # NLI 格式的 negative=1 也丢掉
            elif has_other_label:
                # 同时有其他标签（如 abuse=1, negative=1），保留
                other_samples.append(s)
            else:
                neg1_keep.append(s)
        elif has_other_label:
            # 有其他标签但 negative=0，保留
            other_samples.append(s)
        elif is_nlp_task(s["text"]):
            # negative=0 且是 NLI 格式，丢弃
            neg0_discard += 1
        else:
            # negative=0，无其他标签，非 NLI → 正常文本，保留
            other_samples.append(s)

    print(f"  保留的其他类别/正常样本: {len(other_samples)}")
    print(f"  保留的 negative=1 样本: {len(neg1_keep)}")
    print(f"  丢弃的 negative=0 NLI 样本: {neg0_discard}")
    print(f"  丢弃的 negative=1 NLI 样本: {neg1_discard_nlp}")

    # ===== 3. 加载新的安全文本作为 negative=0 =====
    print("\n" + "=" * 50)
    print("步骤3：加载正面/安全文本替换 negative=0")
    print("=" * 50)

    # 目标：negative=0 的新数据量 ≈ negative=1 的 1.2~1.5 倍
    target_neg0 = int(len(neg1_keep) * 1.3)
    print(f"  目标新增 negative=0 样本: {target_neg0}")

    safe_texts = []

    # 依次尝试加载各个数据源
    weibo_pos = load_weibo_positive(max_count=target_neg0)
    safe_texts.extend(weibo_pos)

    if len(safe_texts) < target_neg0:
        remain = target_neg0 - len(safe_texts)
        chn_pos = load_chnsenticorp_positive(max_count=remain)
        safe_texts.extend(chn_pos)

    if len(safe_texts) < target_neg0:
        remain = target_neg0 - len(safe_texts)
        toutiao = load_toutiao_titles(max_count=remain)
        safe_texts.extend(toutiao)

    if not safe_texts:
        print("\n❌ 没有找到任何安全文本数据集！")
        print("请至少下载以下之一并放到 data/raw/:")
        print("  1. weibo_senti_100k.csv")
        print("     → https://github.com/SophonPlus/ChineseNlpCorpus")
        print("  2. ChnSentiCorp.csv")
        print("     → https://huggingface.co/datasets/seamew/ChnSentiCorp")
        return

    # 控制数量
    if len(safe_texts) > target_neg0:
        safe_texts = random.sample(safe_texts, target_neg0)

    # 转成标准格式
    new_neg0_samples = []
    for text in safe_texts:
        new_neg0_samples.append({
            "text": text,
            "ad": 0,
            "abuse": 0,
            "negative": 0,
            "misinfo": 0,
        })

    print(f"\n  实际新增 negative=0: {len(new_neg0_samples)}")

    # ===== 4. 合并所有数据 =====
    print("\n" + "=" * 50)
    print("步骤4：合并数据")
    print("=" * 50)

    final_data = other_samples + neg1_keep + new_neg0_samples

    # 去重
    seen = set()
    deduped = []
    for s in final_data:
        key = s["text"].strip().lower()[:200]
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    removed_dup = len(final_data) - len(deduped)
    final_data = deduped
    random.shuffle(final_data)

    print(f"  合并后总量: {len(final_data)} (去重 {removed_dup})")

    # ===== 5. 统计最终分布 =====
    print("\n" + "=" * 50)
    print("步骤5：最终数据分布")
    print("=" * 50)

    for lab in LABELS:
        pos = sum(1 for s in final_data if s.get(lab, 0) == 1)
        neg = len(final_data) - pos
        ratio = neg / pos if pos > 0 else float('inf')
        print(f"  {lab}: pos={pos}, neg={neg}, ratio=1:{ratio:.1f}")

    normal = sum(1 for s in final_data
                 if not any([s.get(l, 0) for l in LABELS]))
    print(f"  正常文本(全0): {normal}")

    # ===== 6. 抽样检查 =====
    print("\n=== 抽样：negative=0 新数据 ===")
    for s in random.sample(new_neg0_samples, min(5, len(new_neg0_samples))):
        print(f"  {s['text'][:60]}...")

    print("\n=== 抽样：negative=1 保留数据 ===")
    for s in random.sample(neg1_keep, min(5, len(neg1_keep))):
        print(f"  {s['text'][:60]}...")

    # ===== 7. 划分并保存 =====
    train_split, val_split = train_test_split(
        final_data, test_size=0.15, random_state=42
    )

    save_jsonl(train_split, "data/train.jsonl")
    save_jsonl(val_split, "data/val.jsonl")

    print(f"\n{'=' * 50}")
    print(f"✅ 训练集: {len(train_split)} → data/train.jsonl")
    print(f"✅ 验证集: {len(val_split)} → data/val.jsonl")

    for name, data in [("训练集", train_split), ("验证集", val_split)]:
        print(f"\n  {name}:")
        for lab in LABELS:
            pos = sum(1 for s in data if s.get(lab, 0) == 1)
            print(f"    {lab}: {pos} 正样本")

    print(f"\n下一步: python train.py")


if __name__ == "__main__":
    main()