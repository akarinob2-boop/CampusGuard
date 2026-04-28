# scripts/simulate_user_features.py
"""
为训练数据模拟用户画像特征（轻量版）

【改动说明 v2】
原版问题：违规样本 70% 概率生成"可疑用户"画像，导致模型学会了
         "看画像猜标签"的捷径，而非真正理解文本语义，造成 ad/misinfo
         过度敏感。

修复策略：用户画像完全随机生成，与标签无关。
         模型无法从画像中获得捷径，只能把画像作为辅助信号。
         这样用户画像只会在边界样本上起到微调作用，不会主导决策。

运行: cd E:\ai && python scripts/simulate_user_features.py
"""
import os
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

LABELS = ["ad", "abuse", "negative", "misinfo"]

USER_FEATURE_NAMES = [
    "violation_rate",
    "account_age_days",
    "post_count",
    "avg_daily_posts",
    "recent_violations_7d",
    "is_verified",
    "night_post_ratio",
    "interaction_ratio",
]


def generate_user_features():
    """
    生成随机用户画像特征，与样本标签完全无关。

    各特征按真实校园论坛用户的合理分布采样，
    但不与违规类型挂钩，防止模型走捷径。
    """
    features = {
        # 违规率：大多数用户很低，少数用户偏高（长尾分布）
        "violation_rate": float(np.clip(np.random.beta(1, 8), 0, 1)),

        # 账号年龄：均匀分布，新老用户都有
        "account_age_days": float(np.clip(
            np.random.uniform(1, 1095), 1, 1095) / 1095),

        # 发帖总数：大多数用户发帖不多，少数活跃用户很多
        "post_count": float(np.clip(
            np.random.exponential(150), 1, 2000) / 2000),

        # 日均发帖：大多数用户频率低
        "avg_daily_posts": float(np.clip(
            np.random.exponential(3), 0.1, 50) / 50),

        # 近7天违规数：绝大多数为0
        "recent_violations_7d": float(np.clip(
            np.random.poisson(0.15), 0, 10) / 10),

        # 是否认证：约 25% 的用户认证
        "is_verified": float(random.random() < 0.25),

        # 深夜发帖比：正态分布，均值约 0.2
        "night_post_ratio": float(np.clip(
            np.random.beta(2, 6), 0, 1)),

        # 互动率：中等分布
        "interaction_ratio": float(np.clip(
            np.random.beta(3, 4), 0, 1)),
    }
    return {k: round(v, 4) for k, v in features.items()}


def read_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    for split in ["train", "val"]:
        path = f"data/{split}.jsonl"
        print(f"\n处理 {path}...")

        samples = read_jsonl(path)
        enriched = []

        for s in samples:
            # 随机生成用户画像，与标签无关
            s["user_features"] = generate_user_features()
            enriched.append(s)

        save_jsonl(enriched, path)

        # 统计
        vio_count = sum(1 for s in enriched
                        if any(s.get(l, 0) == 1 for l in LABELS))
        print(f"  总样本: {len(enriched)}")
        print(f"  违规样本: {vio_count}")
        print(f"  正常样本: {len(enriched) - vio_count}")

        # 抽样展示
        sample = random.choice(enriched)
        print(f"  示例: text={sample['text'][:40]}...")
        print(f"         user_features={sample['user_features']}")

    # 保存特征名列表（供 API 使用）
    meta = {
        "feature_names": USER_FEATURE_NAMES,
        "feature_dim": len(USER_FEATURE_NAMES),
    }
    meta_path = "models/user_feature_meta.json"
    os.makedirs("models", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n特征元信息已保存到 {meta_path}")
    print("完成！接下来运行: python train_profile.py")


if __name__ == "__main__":
    main()
