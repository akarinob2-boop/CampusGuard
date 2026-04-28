# api.py — CampusGuard 推理服务（v2 修复版）
"""
【改动说明 v2】
- 修复：原版有两个 FastAPI() 实例，后者覆盖前者，规则层实际失效
- 修复：整合规则层（脏话词典）+ 模型层（语义理解）为单一 app
- 支持用户画像可选传入，无画像时自动降级为纯文本模式
- 返回 profile_influence 替代原 gate_weight，语义更清晰

运行: cd E:\ai && uvicorn api:app --reload --port 8000
"""
import os
import sys
import json
import re
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.campus_guard_model import (
    CampusGuardModel, USER_FEATURE_NAMES, USER_FEATURE_DIM
)

# ==================== 应用初始化 ====================

app = FastAPI(title="CampusGuard API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = ["ad", "abuse", "negative", "misinfo"]
LABEL_CN = {
    "ad": "广告引流",
    "abuse": "辱骂攻击",
    "negative": "消极有害",
    "misinfo": "虚假信息",
}

# 各标签阈值（消极有害调高避免误触发，辱骂/广告调低提升召回）
DEFAULT_THRESHOLDS = {
    "ad":       0.40,
    "abuse":    0.40,
    "negative": 0.65,
    "misinfo":  0.50,
}

# ==================== 请求体定义 ====================

class PredictRequest(BaseModel):
    text: str
    user_features: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "用户画像特征（可选）。"
            "字段：violation_rate, account_age_days, post_count, "
            "avg_daily_posts, recent_violations_7d, is_verified, "
            "night_post_ratio, interaction_ratio"
        ),
        example={
            "violation_rate": 0.1,
            "account_age_days": 0.5,
            "post_count": 0.05,
            "avg_daily_posts": 0.04,
            "recent_violations_7d": 0.0,
            "is_verified": 1.0,
            "night_post_ratio": 0.15,
            "interaction_ratio": 0.4,
        }
    )


# ==================== 脏话规则引擎 ====================

ABUSE_KEYWORDS = {
    # 常见脏话
    "傻逼", "sb", "SB", "煞笔", "傻B", "傻b",
    "操你妈", "草你妈", "日你妈", "艹你妈", "肏你妈",
    "妈逼", "你妈的", "他妈的", "特么的", "尼玛",
    "卧槽", "我操", "我艹", "我草",
    "去死", "该死", "找死", "作死",
    "滚蛋", "滚犊子", "滚出去",
    "废物", "垃圾", "人渣", "败类", "畜生", "禽兽",
    "白痴", "蠢货", "笨蛋", "智障", "脑残", "弱智",
    "贱人", "贱货", "婊子", "荡妇", "骚货", "绿茶婊",
    "屌丝", "屌", "鸡巴", "龟头",
    "狗逼", "狗日的", "狗娘养",
    "混蛋", "王八蛋", "王八",
    "神经病", "有病", "变态",
    "臭逼", "烂逼", "烂货",
    "死全家", "全家死光",
    "fuck", "shit", "bitch", "asshole", "dick",
    # 变体/谐音
    "nmsl", "NMSL", "cnm", "CNM",
    "尼玛死了", "你马死了",
    "沙比", "煞比", "铩比",
    "牛逼", "装逼", "逼格",
    # 新增：威胁/诅咒类
    "我弄死你", "弄死你", "杀了你", "捅死你", "砍死你",
    "打死你", "搞死你", "把你埋了",
    "祖宗十八代", "诅咒你", "活该你死",
    # 新增：侮辱性称谓
    "狗东西", "狗杂种", "狗崽子", "杂种", "野种",
    "蠢猪", "蠢驴", "猪脑子", "狗脑子",
    "loser", "废柴", "社会垃圾",
    # 新增：校园场景常见
    "你全家", "你家人", "滚回去",
    "恶心", "呕吐", "恶心死了",
    "丑八怪", "死肥宅", "死宅",
    "臭不要脸", "不要脸", "无耻",
    "欠打", "欠揍", "活该",
}

ABUSE_PATTERNS = [
    r"[傻煞沙铩][逼比笔B]",
    r"[操草艹肏日]你[妈马麻]",
    r"[你尼][妈马麻][的逼]",
    r"[死去滚].*[吧啊].*[废垃]",
    r"s\s*b",
    r"n\s*m\s*s\s*l",
    r"f\s*u\s*c\s*k",
    r"[打砍捅弄杀][死了]你",
    r"[狗猪蠢][东杂崽脑][西种子袋]",
    r"[丑死肥].*[宅怪鬼]",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in ABUSE_PATTERNS]


# ==================== 广告引流规则引擎 ====================

AD_KEYWORDS = {
    # 引导加联系方式
    "加微信", "加我微信", "加vx", "加VX", "加wx", "加WX",
    "加QQ", "加qq", "私聊", "私信我", "滴滴我",
    "扫码", "扫二维码", "扫我", "长按识别",
    "联系我", "找我", "联系方式",
    # 色情引流
    "看涩图", "看色图", "福利图", "福利视频", "小视频",
    "不雅视频", "私密视频", "约炮", "约p", "约P",
    "一夜情", "援交", "卖身", "包养",
    "可约", "可上门", "可外出",
    # 代购/兼职诈骗
    "日赚", "日入", "月入过万", "轻松赚钱", "躺赚",
    "兼职招聘", "网赚", "刷单", "刷信誉", "拉人头",
    "代购", "低价出", "内部价", "出售账号",
    "免费领取", "限时免费", "抢购",
    # 赌博/违禁
    "网络赌博", "赌球", "线上博彩", "棋牌平台",
    "办证", "办假证", "代开发票", "代办",
    "枪支", "弹药", "毒品", "大麻",
}

AD_PATTERNS = [
    r"加[我]?[微威V][信x]",
    r"v\s*x\s*[:：]?\s*\w+",
    r"wx\s*[:：]?\s*\w+",
    r"qq\s*[:：]?\s*\d+",
    r"日[赚入]\s*\d+",
    r"月[赚入]\s*\d+",
    r"[可私]约",
    r"[扫长].*[码识别]",
    r"\d{5,}.*[元块].*[赚入]",
]

_compiled_ad_patterns = [re.compile(p, re.IGNORECASE) for p in AD_PATTERNS]


def check_abuse_keywords(text: str) -> dict:
    text_lower = text.lower()
    matched = []

    for word in ABUSE_KEYWORDS:
        if word.lower() in text_lower:
            matched.append(word)

    if matched:
        return {"hit": True, "matched_words": matched, "method": "keyword"}

    for pattern in _compiled_patterns:
        m = pattern.search(text)
        if m:
            matched.append(m.group())

    if matched:
        return {"hit": True, "matched_words": matched, "method": "pattern"}

    return {"hit": False, "matched_words": [], "method": None}


def check_ad_keywords(text: str) -> dict:
    text_lower = text.lower()
    matched = []

    for word in AD_KEYWORDS:
        if word.lower() in text_lower:
            matched.append(word)

    if matched:
        return {"hit": True, "matched_words": matched, "method": "keyword"}

    for pattern in _compiled_ad_patterns:
        m = pattern.search(text)
        if m:
            matched.append(m.group())

    if matched:
        return {"hit": True, "matched_words": matched, "method": "pattern"}

    return {"hit": False, "matched_words": [], "method": None}


# ==================== 模型加载 ====================

MODEL_DIR = "./models/campus_guard"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("正在加载 CampusGuard 融合模型...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = CampusGuardModel.from_pretrained(MODEL_DIR, device=DEVICE)
model.eval()
print(f"模型加载完成！设备: {DEVICE}")

# 加载阈值（优先读文件，否则用代码默认值）
THRESHOLDS = DEFAULT_THRESHOLDS.copy()
th_path = os.path.join(MODEL_DIR, "thresholds.json")
if os.path.exists(th_path):
    with open(th_path, "r", encoding="utf-8") as f:
        th_data = json.load(f)
        THRESHOLDS.update(th_data.get("thresholds", {}))
print(f"阈值: {THRESHOLDS}")


# ==================== 推理接口 ====================

@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text

    # ===== 第1层：规则检测（脏话词典 + 广告词典，零延迟）=====
    abuse_check = check_abuse_keywords(text)
    ad_check = check_ad_keywords(text)

    # ===== 第2层：模型检测（语义理解）=====
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=256, padding="max_length",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 用户画像特征（可选）
    user_feat_tensor = None
    if request.user_features:
        feat_values = [
            float(request.user_features.get(fn, 0.0))
            for fn in USER_FEATURE_NAMES
        ]
        user_feat_tensor = torch.tensor(
            [feat_values], dtype=torch.float32
        ).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
            user_features=user_feat_tensor,
        )
        logits = outputs["logits"]
        probs = torch.sigmoid(logits).squeeze().tolist()
        profile_influence = outputs["profile_influence"]  # 画像实际影响幅度

    if not isinstance(probs, list):
        probs = [probs]

    # ===== 合并规则层与模型层结果 =====
    result_labels = {}
    is_violation = False

    for i, lab in enumerate(LABELS):
        prob = probs[i]
        threshold = THRESHOLDS.get(lab, 0.5)

        # abuse：规则层命中则强制触发
        if lab == "abuse" and abuse_check["hit"]:
            is_triggered = True
            prob = max(prob, 0.99)
            trigger_source = "rule+model" if probs[i] >= threshold else "rule"
        # ad：规则层命中则强制触发
        elif lab == "ad" and ad_check["hit"]:
            is_triggered = True
            prob = max(prob, 0.99)
            trigger_source = "rule+model" if probs[i] >= threshold else "rule"
        else:
            is_triggered = prob >= threshold
            trigger_source = "model" if is_triggered else None

        result_labels[lab] = {
            "label_cn": LABEL_CN.get(lab, lab),
            "probability": round(prob, 4),
            "threshold": threshold,
            "is_triggered": is_triggered,
            "source": trigger_source,
        }

        if is_triggered:
            is_violation = True

    # ===== 构建响应 =====
    response = {
        "text": text,
        "is_violation": is_violation,
        "details": result_labels,
        "model_info": {
            "version": "2.1-profile-bias",
            "has_user_features": request.user_features is not None,
            # profile_influence：画像对本次预测的调节幅度
            # 0.0 = 纯文本模式，0.3 = 画像最大影响
            "profile_influence": round(profile_influence, 4),
            "profile_interpretation": (
                "纯文本模式（无用户画像）" if not request.user_features
                else f"用户画像调节幅度 {profile_influence:.1%}"
            ),
        },
    }

    # 附加规则层信息
    if abuse_check["hit"]:
        response["rule_info"] = {
            "abuse_keywords_hit": True,
            "matched_words": abuse_check["matched_words"],
            "match_method": abuse_check["method"],
        }
    if ad_check["hit"]:
        response.setdefault("rule_info", {})
        response["rule_info"].update({
            "ad_keywords_hit": True,
            "ad_matched_words": ad_check["matched_words"],
            "ad_match_method": ad_check["method"],
        })

    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": "2.1-profile-bias",
        "device": DEVICE,
        "thresholds": THRESHOLDS,
    }
