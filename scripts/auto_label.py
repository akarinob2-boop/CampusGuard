# -*- coding: utf-8 -*-
"""
auto_label.py — 调用本地推理 API 对爬取数据自动打标
用法：
    # 先确保 api 服务已启动：uvicorn api:app --port 8000
    python scripts/auto_label.py --inp data/tieba_raw.jsonl --out_auto data/tieba_labeled.jsonl --out_review data/tieba_review.jsonl
"""
import argparse
import json
import time
import requests

API_URL = "http://localhost:8000/predict"

# 置信度阈值：概率高于 AUTO_HIGH → 自动标 1，低于 AUTO_LOW → 自动标 0，中间送人工复核
AUTO_HIGH = 0.80
AUTO_LOW  = 0.20


def predict(text: str) -> dict | None:
    try:
        resp = requests.post(API_URL, json={"text": text}, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  [error] 请求失败: {e}")
    return None


def auto_label(details: dict) -> tuple[dict, bool]:
    """
    根据模型置信度决定标签。
    返回 (labels_dict, need_review)
      - labels_dict: {"ad": 0/1, "abuse": 0/1, ...}
      - need_review: True 表示至少有一个标签需要人工复核
    """
    labels = {}
    need_review = False

    for lab in ["ad", "abuse", "negative", "misinfo"]:
        prob = details[lab]["probability"]
        if prob >= AUTO_HIGH:
            labels[lab] = 1
        elif prob <= AUTO_LOW:
            labels[lab] = 0
        else:
            labels[lab] = -1   # 不确定，送复核
            need_review = True

    return labels, need_review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp",        default="data/tieba_raw.jsonl",      help="输入文件（爬虫输出）")
    parser.add_argument("--out_auto",   default="data/tieba_labeled.jsonl",   help="高置信度自动打标结果")
    parser.add_argument("--out_review", default="data/tieba_review.jsonl",    help="需人工复核的样本")
    parser.add_argument("--delay",      type=float, default=0.05,             help="请求间隔秒数")
    args = parser.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"共 {len(items)} 条待打标，AUTO_HIGH={AUTO_HIGH}, AUTO_LOW={AUTO_LOW}")

    auto_rows   = []
    review_rows = []
    fail_count  = 0

    for i, item in enumerate(items):
        text = item.get("text", "").strip()
        if not text:
            continue

        result = predict(text)
        if result is None:
            fail_count += 1
            continue

        labels, need_review = auto_label(result["details"])

        row = {
            "text":    text,
            "ad":      labels["ad"],
            "abuse":   labels["abuse"],
            "negative":labels["negative"],
            "misinfo": labels["misinfo"],
            # 保留模型原始概率，方便复核时参考
            "_probs": {
                lab: round(result["details"][lab]["probability"], 4)
                for lab in ["ad", "abuse", "negative", "misinfo"]
            },
            "_source": item.get("source", ""),
        }

        if need_review:
            review_rows.append(row)
        else:
            # 自动打标结果去掉辅助字段，对齐训练集格式
            clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
            auto_rows.append(clean_row)

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(items)} | 自动:{len(auto_rows)} 复核:{len(review_rows)} 失败:{fail_count}")

        time.sleep(args.delay)

    # 写出结果
    with open(args.out_auto, "w", encoding="utf-8") as f:
        for row in auto_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.out_review, "w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"""
完成！
  自动打标（直接可用）: {len(auto_rows)} 条  → {args.out_auto}
  需人工复核:          {len(review_rows)} 条  → {args.out_review}
  请求失败:            {fail_count} 条
""")


if __name__ == "__main__":
    main()
