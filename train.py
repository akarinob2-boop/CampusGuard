# train.py — 改进版 v3（修复所有兼容性问题）
import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, precision_score, recall_score

LABELS = ["ad", "abuse", "negative", "misinfo"]

MODEL_REPO = "hfl/chinese-macbert-base"
LOCAL_MODEL_DIR = "models/hf_cache/chinese-macbert-base"


# ==================== FGM 对抗训练 ====================
class FGM:
    """Fast Gradient Method：在 embedding 层添加对抗扰动"""

    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# ==================== 自定义 Trainer ====================
class CampusTrainer(Trainer):
    """
    自定义 Trainer，集成：
    1. Focal Loss（聚焦困难样本，γ=2）
    2. pos_weight（自动补偿少数类）
    3. FGM 对抗训练（提升鲁棒性）
    """

    def __init__(self, pos_weight=None, focal_gamma=2.0,
                 use_fgm=True, fgm_epsilon=1.0, **kwargs):
        super().__init__(**kwargs)
        self.focal_gamma = focal_gamma
        self.use_fgm = use_fgm
        self.fgm_epsilon = fgm_epsilon

        if pos_weight is not None:
            self._pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        else:
            self._pos_weight = None

        self.fgm = None

    # ---------- 核心：自定义损失函数 ----------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ★ 加了 **kwargs 兼容新版 transformers 传入 num_items_in_batch 等额外参数
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # --- Focal Loss ---
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        p_t = labels * probs + (1 - labels) * (1 - probs)
        focal_weight = (1 - p_t) ** self.focal_gamma
        loss = focal_weight * bce

        # --- pos_weight：正样本乘以更大的权重 ---
        if self._pos_weight is not None:
            pw = self._pos_weight.to(logits.device)
            sample_weight = labels * pw.unsqueeze(0) + (1 - labels) * 1.0
            loss = loss * sample_weight

        loss = loss.mean()

        # ★ 把 labels 放回去，防止后续流程需要
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss

    # ---------- FGM 对抗训练 ----------
    def training_step(self, model, inputs, num_items_in_batch=None):
        # ★ 修复：添加 num_items_in_batch 参数，兼容新版 transformers
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 初始化 FGM（只做一次）
        if self.use_fgm and self.fgm is None:
            self.fgm = FGM(model, epsilon=self.fgm_epsilon)

        # ① 正常 forward + backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # 使用 accelerator 做 backward（兼容 fp16/DeepSpeed 等）
        self.accelerator.backward(loss)

        # ② FGM 对抗训练：在 embedding 层加扰动，再 forward + backward
        if self.use_fgm and self.fgm is not None:
            self.fgm.attack()
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv)
            self.fgm.restore()

        return loss.detach() / self.args.gradient_accumulation_steps


# ==================== 指标计算 ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {}
    metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
    metrics["micro_f1"] = f1_score(labels, preds, average="micro", zero_division=0)
    for i, lab in enumerate(LABELS):
        metrics[f"{lab}_f1"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
        metrics[f"{lab}_p"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
        metrics[f"{lab}_r"] = recall_score(labels[:, i], preds[:, i], zero_division=0)
    return metrics


# ==================== 模型下载 ====================
def ensure_local_model():
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN", None),
            resume_download=True,
        )
    return LOCAL_MODEL_DIR


# ==================== 自动计算 pos_weight ====================
def calc_pos_weight(dataset, labels):
    n = len(dataset)
    weights = []
    for i, lab in enumerate(labels):
        pos_count = sum(1 for x in dataset if x["labels"][i] > 0.5)
        neg_count = n - pos_count
        if pos_count == 0:
            w = 1.0
        else:
            w = neg_count / pos_count
        w = min(w, 20.0)
        weights.append(round(w, 2))
        print(f"  {lab}: pos={pos_count}, neg={neg_count}, pos_weight={weights[-1]}")
    return weights


# ==================== 训练参数 ====================
def build_training_args():
    sig = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs = {
        "output_dir": "runs/macbert_multilabel",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 8,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "gradient_accumulation_steps": 2,
    }

    # ★ warmup：优先用 warmup_steps（新版），否则用 warmup_ratio（旧版）
    if "warmup_steps" in sig:
        kwargs["warmup_steps"] = 200
    elif "warmup_ratio" in sig:
        kwargs["warmup_ratio"] = 0.1

    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in sig:
        kwargs["save_strategy"] = "epoch"
    if "logging_strategy" in sig:
        kwargs["logging_strategy"] = "steps"
    if "load_best_model_at_end" in sig:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in sig:
        kwargs["metric_for_best_model"] = "macro_f1"
    if "greater_is_better" in sig:
        kwargs["greater_is_better"] = True
    if "fp16" in sig:
        kwargs["fp16"] = torch.cuda.is_available()
    if "dataloader_pin_memory" in sig:
        kwargs["dataloader_pin_memory"] = torch.cuda.is_available()
    if "report_to" in sig:
        kwargs["report_to"] = "none"
    if "save_safetensors" in sig:
        kwargs["save_safetensors"] = True
    if "save_total_limit" in sig:
        kwargs["save_total_limit"] = 3

    return TrainingArguments(**kwargs)


# ==================== 主训练流程 ====================
def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_model = ensure_local_model()
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)

    # ---------- 加载数据 ----------
    ds = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"},
    )

    # ★ 修复：先获取原始列名，然后再做所有处理
    original_columns = ds["train"].column_names
    print(f"原始列名: {original_columns}")
    print(f"训练集大小: {len(ds['train'])}, 验证集大小: {len(ds['val'])}")

    # ★ 安全过滤：移除 text 为空的样本
    def is_valid(example):
        text = example.get("text", None)
        return text is not None and isinstance(text, str) and len(text.strip()) > 0

    ds = ds.filter(is_valid)
    print(f"过滤后 — 训练集: {len(ds['train'])}, 验证集: {len(ds['val'])}")

    # ★ 修复：preprocess 函数中增加安全检查
    def preprocess(example):
        text = str(example["text"]).strip()
        enc = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=False,
        )
        enc["labels"] = [float(example.get(l, 0)) for l in LABELS]
        return enc

    # ★ 修复：用之前保存的 original_columns 来 remove，而非重新取 column_names
    columns_to_remove = [c for c in original_columns if c in ds["train"].column_names]
    ds = ds.map(preprocess, remove_columns=columns_to_remove)

    # ---------- 加载模型 ----------
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model,
        local_files_only=True,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    # ---------- 计算 pos_weight ----------
    print("\n=== 计算类别权重 ===")
    pos_weight = calc_pos_weight(ds["train"], LABELS)
    print(f"pos_weight = {pos_weight}\n")

    # ---------- 开始训练 ----------
    args = build_training_args()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CampusTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        pos_weight=pos_weight,
        focal_gamma=2.0,
        use_fgm=True,
        fgm_epsilon=1.0,
    )

    trainer.train()

    # ---------- 保存 ----------
    save_dir = "models/campus_detector"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n训练完成，已保存到 {save_dir}")
    print("请接下来运行: python scripts/tune_thresholds.py 来搜索最佳阈值")


if __name__ == "__main__":
    main()
