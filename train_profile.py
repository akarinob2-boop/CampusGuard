# train_profile.py
"""
CampusGuard 融合模型训练脚本
集成：Focal Loss + pos_weight + FGM 对抗训练 + 用户画像融合

运行: cd E:\ai && python train_profile.py
"""
import os
import json
import inspect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, precision_score, recall_score

# 导入自定义模型
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.campus_guard_model import (
    CampusGuardModel, USER_FEATURE_NAMES, USER_FEATURE_DIM
)

LABELS = ["ad", "abuse", "negative", "misinfo"]
LOCAL_MODEL_DIR = "models/hf_cache/chinese-macbert-base"


# ==================== 数据集 ====================

class CampusDataset(Dataset):
    """支持用户画像特征的数据集"""

    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 文本编码
        text = str(sample["text"]).strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        # 用户画像特征
        user_feats = sample.get("user_features", None)
        if user_feats:
            feat_values = [float(user_feats.get(fn, 0.0))
                           for fn in USER_FEATURE_NAMES]
            item["user_features"] = torch.tensor(feat_values, dtype=torch.float32)
        else:
            item["user_features"] = torch.zeros(USER_FEATURE_DIM, dtype=torch.float32)

        # 标签
        item["labels"] = torch.tensor(
            [float(sample.get(l, 0)) for l in LABELS],
            dtype=torch.float32
        )

        return item


# ==================== GCLoss（梯度约束损失）====================

class GCLoss(nn.Module):
    """
    Gradient-Constrained Loss（来自 ToxiTrace）

    惩罚模型在 PAD token 上的梯度显著性，
    迫使模型只关注真实内容 token，而非填充位置的伪相关性。

    实现方式：
      1. 注册 forward hook 捕获 embedding 层输出
      2. 对 CE loss 反向传播，获取 embedding 梯度
      3. 计算每个 token 的 L2 显著性分数
      4. 对 PAD 位置（attention_mask=0）的显著性求均值作为惩罚项

    参考：ToxiTrace (arXiv:2604.12321) §3.2 GCLoss
    """

    def __init__(self):
        super().__init__()
        self._emb_output = None
        self._hook_handle = None

    def _hook_fn(self, module, input, output):
        # 保存 embedding 输出的引用（训练时 requires_grad=True）
        self._emb_output = output

    def register(self, bert_embeddings: nn.Module):
        """在 bert.embeddings 上注册 hook"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = bert_embeddings.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def forward(self, logits, labels, attention_mask):
        """
        Args:
            logits:          (batch, num_labels)
            labels:          (batch, num_labels) float
            attention_mask:  (batch, seq_len)  1=真实token, 0=PAD
        Returns:
            gc_loss: scalar（可直接加入主 loss 反向传播）

        实现说明：
        PyTorch 的 FlashAttention / efficient_attention 不支持二阶导数，
        因此无法用 create_graph=True。改为：
          1. 用 detach 后的梯度计算 PAD 显著性权重（无法回传，仅作系数）
          2. 用该权重对 embedding 的 L2 范数加权求和，构造可回传的标量
          3. 这样梯度只流经 embedding→attention→logits 的一阶路径，
             同时惩罚了 PAD token 对损失的贡献
        """
        if self._emb_output is None or not self._emb_output.requires_grad:
            return torch.tensor(0.0, device=logits.device)

        emb = self._emb_output  # (batch, seq_len, hidden)

        # ── Step 1: 用 detach 版本的梯度计算显著性权重（不走二阶图）──
        with torch.no_grad():
            aux_loss = nn.functional.binary_cross_entropy_with_logits(
                logits.detach(), labels, reduction="mean"
            )
        # 对 embedding 求一阶梯度，detach 断开二阶路径
        try:
            grads = torch.autograd.grad(
                nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="mean"
                ),
                emb,
                create_graph=False,   # 不建二阶图，兼容 FlashAttention
                retain_graph=True,
            )[0].detach()  # (batch, seq_len, hidden)  断开梯度链
        except RuntimeError:
            # 万一仍不支持，直接跳过 GCLoss 不影响主训练
            return torch.tensor(0.0, device=logits.device)

        # 每个 token 的 L2 显著性（已 detach，作为常数权重）
        saliency_weight = grads.norm(dim=-1)  # (batch, seq_len)

        # PAD 位置掩码
        pad_mask = (attention_mask == 0).float()  # (batch, seq_len)

        # ── Step 2: 用显著性权重 × embedding L2 范数 构造可回传的惩罚项 ──
        # embedding 的 L2 范数本身是可微的，乘以常数权重后梯度照常流回 BERT
        emb_norm = emb.norm(dim=-1)  # (batch, seq_len)

        pad_penalty = (saliency_weight * emb_norm * pad_mask).sum(dim=-1)  # (batch,)
        pad_count = pad_mask.sum(dim=-1).clamp(min=1)
        gc_loss = (pad_penalty / pad_count).mean()

        return gc_loss


# ==================== ARCLoss（自适应推理对比学习）====================

class ARCLoss(nn.Module):
    """
    Adaptive Reasoning Contrastive Learning（来自 ToxiTrace）

    对同一批次进行两次前向传播（Dropout 随机性产生不同的 dropout mask），
    将同一样本的两次 [CLS] 表示视为正样本对，
    其他样本视为负样本，用 InfoNCE 损失拉近正对、推远负对。

    这迫使模型学习对 dropout 扰动鲁棒的语义表示，
    类似于 SimCSE 的无监督对比学习。

    参考：ToxiTrace (arXiv:2604.12321) §3.3 ARCL
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: (batch, hidden)  第一次前向传播的 [CLS] 表示
            z2: (batch, hidden)  第二次前向传播的 [CLS] 表示
        Returns:
            arcl_loss: scalar
        """
        batch_size = z1.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z1.device)

        # L2 归一化
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)

        # 相似度矩阵 (batch, batch)
        sim = torch.matmul(z1, z2.T) / self.temperature

        # 对角线为正样本对
        labels = torch.arange(batch_size, device=z1.device)

        # 双向 InfoNCE
        loss = (
            nn.functional.cross_entropy(sim, labels) +
            nn.functional.cross_entropy(sim.T, labels)
        ) / 2.0

        return loss


# ==================== FGM 对抗训练 ====================

class FGM:
    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is not None:
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

class ProfileTrainer(Trainer):
    """
    支持用户画像融合的 Trainer
    集成 Focal Loss + pos_weight + FGM
    """

    def __init__(self, pos_weight=None, focal_gamma=2.0,
                 use_fgm=True, fgm_epsilon=1.0,
                 use_gc_loss=True, gc_alpha=0.05,
                 use_arcl=True, arcl_beta=0.05, arcl_temperature=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.focal_gamma = focal_gamma
        self.use_fgm = use_fgm
        self.fgm_epsilon = fgm_epsilon
        self.use_gc_loss = use_gc_loss
        self.gc_alpha = gc_alpha
        self.use_arcl = use_arcl
        self.arcl_beta = arcl_beta

        if pos_weight is not None:
            self._pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
        else:
            self._pos_weight = None

        self.fgm = None
        self._gc_loss_fn = GCLoss() if use_gc_loss else None
        self._arcl_loss_fn = ARCLoss(temperature=arcl_temperature) if use_arcl else None
        self._gc_registered = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        user_features = inputs.pop("user_features", None)
        attention_mask = inputs.get("attention_mask")

        # ---- GCLoss：注册 embedding hook（只注册一次）----
        if self.use_gc_loss and self._gc_loss_fn is not None and not self._gc_registered:
            # 找到真实的 bert 模块（可能被 DataParallel 包裹）
            bert_module = model.module if hasattr(model, "module") else model
            self._gc_loss_fn.register(bert_module.bert.embeddings)
            self._gc_registered = True

        # ---- 第一次前向传播 ----
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            token_type_ids=inputs.get("token_type_ids"),
            user_features=user_features,
        )
        logits = outputs["logits"]
        cls_emb_1 = outputs.get("cls_emb")  # (batch, 768)

        # ---- Focal Loss ----
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        p_t = labels * probs + (1 - labels) * (1 - probs)
        focal_weight = (1 - p_t) ** self.focal_gamma
        loss = focal_weight * bce

        # pos_weight
        if self._pos_weight is not None:
            pw = self._pos_weight.to(logits.device)
            sample_weight = labels * pw.unsqueeze(0) + (1 - labels) * 1.0
            loss = loss * sample_weight

        focal_loss = loss.mean()
        total_loss = focal_loss

        # ---- GCLoss ----
        if self.use_gc_loss and self._gc_loss_fn is not None and attention_mask is not None:
            gc_loss = self._gc_loss_fn(logits, labels, attention_mask)
            total_loss = total_loss + self.gc_alpha * gc_loss

        # ---- ARCL：第二次前向传播（不同 dropout mask）----
        if self.use_arcl and self._arcl_loss_fn is not None and cls_emb_1 is not None:
            model.train()  # 确保 dropout 激活
            outputs_2 = model(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
                token_type_ids=inputs.get("token_type_ids"),
                user_features=user_features,
            )
            cls_emb_2 = outputs_2.get("cls_emb")
            if cls_emb_2 is not None:
                arcl_loss = self._arcl_loss_fn(cls_emb_1, cls_emb_2)
                total_loss = total_loss + self.arcl_beta * arcl_loss

        inputs["labels"] = labels
        if user_features is not None:
            inputs["user_features"] = user_features

        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_fgm and self.fgm is None:
            self.fgm = FGM(model, epsilon=self.fgm_epsilon)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        if self.use_fgm and self.fgm is not None:
            self.fgm.attack()
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            self.accelerator.backward(loss_adv)
            self.fgm.restore()

        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        """重写预测步骤，确保 user_features 正确传入"""
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        user_features = inputs.pop("user_features", None)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                user_features=user_features,
            )
            logits = outputs["logits"]

            # 计算 loss
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="mean"
            )

        if prediction_loss_only:
            return (bce, None, None)

        return (bce, logits, labels)

    def _save(self, output_dir=None, state_dict=None):
        """重写 _save 方法，确保所有张量在保存前是连续的(contiguous)"""
        if state_dict is None:
            state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and not value.is_contiguous():
                state_dict[key] = value.contiguous()
        super()._save(output_dir, state_dict)


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


# ==================== pos_weight 计算 ====================

def calc_pos_weight(dataset, labels):
    n = len(dataset)
    weights = []
    for i, lab in enumerate(labels):
        pos_count = sum(1 for j in range(n)
                        if dataset[j]["labels"][i].item() > 0.5)
        neg_count = n - pos_count
        w = neg_count / pos_count if pos_count > 0 else 1.0
        w = min(w, 20.0)
        weights.append(round(w, 2))
        print(f"  {lab}: pos={pos_count}, neg={neg_count}, pos_weight={weights[-1]}")
    return weights


# ==================== 训练参数 ====================

def build_training_args():
    sig = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs = {
        "output_dir": "runs/campus_guard_profile",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 8,
        "weight_decay": 0.01,
        "logging_steps": 50,
        "gradient_accumulation_steps": 2,
    }

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
    if "save_total_limit" in sig:
        kwargs["save_total_limit"] = 3

    return TrainingArguments(**kwargs)


# ==================== 差异化学习率 ====================

def get_optimizer_grouped_parameters(model, bert_lr=2e-5, other_lr=1e-3):
    """
    差异化学习率：
    - BERT 参数：小学习率（微调，保留预训练知识）
    - 分类头：中等学习率（从零学习）
    - 用户画像模块（user_bias）：极小学习率（0.1×other_lr）
      防止画像模块在训练初期过快收敛并主导预测

    【v2 改动】user_bias 单独设置更小的学习率，
    配合 profile_scale 初始值 0.05，确保模型先学好文本再利用画像。
    """
    bert_params = []
    profile_params = []   # user_bias 模块参数
    other_params = []     # 分类头等其他新增参数

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bert" in name:
            bert_params.append(param)
        elif "user_bias" in name:
            profile_params.append(param)
        else:
            other_params.append(param)

    profile_lr = other_lr * 0.1   # 画像模块 lr 降为 1e-4

    print(f"  BERT参数量:    {sum(p.numel() for p in bert_params):,} (lr={bert_lr})")
    print(f"  分类头参数量:  {sum(p.numel() for p in other_params):,} (lr={other_lr})")
    print(f"  画像模块参数量:{sum(p.numel() for p in profile_params):,} (lr={profile_lr})")

    return [
        {"params": bert_params,    "lr": bert_lr},
        {"params": other_params,   "lr": other_lr},
        {"params": profile_params, "lr": profile_lr},
    ]


# ==================== 主训练流程 ====================

def main():
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)

    # ---- 加载数据 ----
    print("=== 加载数据 ===")
    train_dataset = CampusDataset("data/train.jsonl", tokenizer)
    val_dataset = CampusDataset("data/val.jsonl", tokenizer)
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")

    # 检查是否有 user_features
    sample = train_dataset[0]
    has_user = sample["user_features"].sum().item() > 0
    print(f"用户画像特征: {'✅ 已加载' if has_user else '❌ 全为零'}")
    if not has_user:
        print("⚠️  请先运行: python scripts/simulate_user_features.py")
        return

    # ---- 创建模型 ----
    print("\n=== 创建融合模型 ===")
    model = CampusGuardModel(
        bert_path=LOCAL_MODEL_DIR,
        num_labels=len(LABELS),
        user_feature_dim=USER_FEATURE_DIM,
        user_hidden_dim=32,
        user_output_dim=64,
        fusion_output_dim=256,
        dropout=0.1,
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 修复 transformers Trainer 访问 model.config.use_cache 报错的问题
    if hasattr(model, "config"):
        if isinstance(model.config, dict):
            from types import SimpleNamespace
            model.config = SimpleNamespace(**model.config)
        # 确保存在 use_cache 属性
        if not hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    else:
        from types import SimpleNamespace
        model.config = SimpleNamespace(use_cache=False)

    # ---- pos_weight ----
    print("\n=== 计算类别权重 ===")
    pos_weight = calc_pos_weight(train_dataset, LABELS)
    print(f"pos_weight = {pos_weight}\n")

    # ---- 差异化学习率优化器 ----
    print("=== 差异化学习率 ===")
    optimizer_groups = get_optimizer_grouped_parameters(
        model, bert_lr=2e-5, other_lr=1e-3
    )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)

    # ---- 训练参数 ----
    args = build_training_args()

    # ---- Trainer ----
    trainer = ProfileTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(optimizer, None),  # 使用自定义优化器
        pos_weight=pos_weight,
        focal_gamma=2.0,
        use_fgm=True,
        fgm_epsilon=1.0,
        # ToxiTrace 辅助损失
        use_gc_loss=True,
        gc_alpha=0.05,          # GCLoss 权重：惩罚 PAD 显著性
        use_arcl=True,
        arcl_beta=0.05,         # ARCL 权重：对比学习
        arcl_temperature=0.1,   # InfoNCE 温度
    )

    # ---- 训练 ----
    print("\n=== 开始训练 ===")
    trainer.train()

    # ---- 保存 ----
    save_dir = "models/campus_guard"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n融合模型已保存到 {save_dir}")
    print("接下来请运行: python scripts/tune_thresholds_profile.py")


if __name__ == "__main__":
    main()
