"""
《猫语心声》 —— LLM集成配置

模型: DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF
推理框架: llama-cpp-python
量化: Q4_0 (4-bit, ~1.0GB显存/内存)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════
#  模型路径
# ═══════════════════════════════════════════

MODEL_DIR = os.path.join(
    BASE_DIR, "llm_integration",
    "DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF"
)
MODEL_FILENAME = "deepseek-r1-distill-qwen-1.5b-q4_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# ═══════════════════════════════════════════
#  llama.cpp 服务器配置
# ═══════════════════════════════════════════

LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_CTX_SIZE = 2048       # 上下文窗口（心声生成仅需~500 tokens）
LLAMA_SERVER_N_GPU_LAYERS = 0      # CPU推理 (Windows无CUDA)
LLAMA_SERVER_THREADS = 8           # CPU线程数
LLAMA_SERVER_BATCH_SIZE = 512

# ═══════════════════════════════════════════
#  llama-cpp-python 本地推理配置
# ═══════════════════════════════════════════

LLAMA_CPP_N_CTX = 2048
LLAMA_CPP_N_THREADS = 8
LLAMA_CPP_N_BATCH = 512
LLAMA_CPP_MAX_TOKENS = 64          # 心声最多64 tokens (~30中文字)
LLAMA_CPP_TEMPERATURE = 0.7
LLAMA_CPP_TOP_P = 0.9
LLAMA_CPP_TOP_K = 40
LLAMA_CPP_REPEAT_PENALTY = 1.1
LLAMA_CPP_STOP = ["<|endoftext|>", "<|im_end|>", "\n\n", "。\n"]

# ═══════════════════════════════════════════
#  缓存配置
# ═══════════════════════════════════════════

CACHE_TTL_SECONDS = 30.0           # 同一猫咪同意图情绪桶30秒内不重复生成
CACHE_MAX_SIZE = 512               # LRU缓存最大条目
EMOTION_BUCKET_SIZE = 20           # 情绪值分桶粒度 (0-20, 20-40, ..., 80-100)

# ═══════════════════════════════════════════
#  降级配置
# ═══════════════════════════════════════════

LLM_TIMEOUT_MS = 150               # LLM推理超时阈值（ms）
FALLBACK_COOLDOWN_SECONDS = 5.0    # LLM故障后冷却期
MAX_CONSECUTIVE_FAILURES = 3       # 连续失败N次后强制使用模板
LLM_HEALTH_CHECK_INTERVAL = 10.0   # 健康检查间隔（秒）

# ═══════════════════════════════════════════
#  并发控制
# ═══════════════════════════════════════════

MAX_CONCURRENT_GENERATIONS = 3     # 同时最多生成3条心声
GENERATION_QUEUE_SIZE = 10         # 排队上限

# ═══════════════════════════════════════════
#  提示词配置
# ═══════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """你是一只名叫{name}的{breed}猫咪。
你的性格：{personality_desc}。
你最重要的一条行为准则：你{personality_core_rule}
你的过往：{backstory}

说话规则：
1. 用第一人称内心独白，不超过20字
2. 语气必须{tone_requirement}
3. 不要解释、不要旁白、不要思考过程，直接输出独白文本
4. 只输出一句内心独白，不要带引号"""

USER_PROMPT_TEMPLATE = """当前状态：
- 情绪：饥饿{hunger}%, 恐惧{fear}%, 好奇{curiosity}%, 舒适{comfort}%, 社交需求{social}%
- 对玩家的信任度：{trust}/100
- 当前正在：{intent_desc}
- 玩家正在：{player_action_desc}

相关记忆：
{memory_context}

当前场景：{scene_desc}。时间：{time_desc}。

你此刻心里在想什么？"""

# ═══════════════════════════════════════════
#  性格语气模板
# ═══════════════════════════════════════════

PERSONALITY_TONE = {
    "傲娇": "嘴硬心软，表面嫌弃但内心在意，用'哼'、'才不是'、'随便你'等口是心非的表达",
    "怯懦": "小心翼翼，充满不安和试探，多用省略号、疑问句，语气轻而犹豫",
    "好奇": "对一切充满探索欲，语气活泼跳跃，爱用'咦'、'这是什么'",
    "贪吃": "以食物为第一关注点，开口闭口和吃的有关，简单直接",
    "社交": "温暖友好，喜欢撒娇、分享，语气甜而亲近",
    "攻击性": "警惕、不信任，语气生硬、简短，带有领地意识",
    "活跃": "精力充沛，语气跳跃、兴奋，一分钟停不下来",
    "独立": "我行我素，语气淡定、从容，不需要讨好任何人",
}

PERSONALITY_CORE_RULE = {
    "傲娇": "总是嘴硬心软，哪怕心里很在意也绝不说出口",
    "怯懦": "极度缺乏安全感，对任何风吹草动都保持高度警惕",
    "好奇": "对一切新鲜事物都有强烈的探索欲望",
    "贪吃": "食物永远是你最关心的事情",
    "社交": "渴望被关注和爱，喜欢与人和猫亲近",
    "攻击性": "对威胁保持高度警惕，随时准备捍卫自己的领地",
    "活跃": "精力充沛，一刻也停不下来",
    "独立": "我行我素，享受独处，不需要刻意讨好任何人",
}

# 意图→自然语言描述
INTENT_DESCRIPTIONS = {
    "idle_wander": "漫无目的地闲逛",
    "approach_player": "慢慢靠近玩家",
    "ask_for_attention": "蹭玩家的腿，想被关注",
    "eat": "低头吃东西",
    "sleep": "蜷缩着打盹",
    "play_with_toy": "追着玩具跑来跑去",
    "social_groom": "和另一只猫互相舔毛",
    "social_play": "和另一只猫追逐打闹",
    "hide": "躲在隐蔽的角落里",
    "hiss_warning": "发出低沉的哈气声",
    "curious_inspect": "好奇地打量周围的东西",
    "follow_player": "跟在玩家身后",
    "accept_petting": "眯着眼享受玩家的抚摸",
    "fearful_retreat": "胆怯地往后退",
    "stare_at_window": "安静地望着窗外发呆",
}

# 玩家行为→自然语言描述
PLAYER_ACTION_DESCRIPTIONS = {
    "pet": "在轻轻抚摸你",
    "feed": "往食盆里倒猫粮",
    "call": "在叫你的名字",
    "play": "拿着逗猫棒在逗你",
    "ignore": "在忙自己的事情",
    "scold": "正在用责备的语气说话",
    "approach": "正在朝你走过来",
    "leave": "转身离开了房间",
    "treat": "手里拿着零食在引诱你",
    "heal": "正在仔细查看你的身体状况",
    "photo": "举起相机在给你拍照",
    "none": "安静地待在房间里",
    "soothe": "蹲下来用很轻的声音在安抚你",
}

# 时间描述
TIME_DESCRIPTIONS = {
    "morning": "清晨的阳光刚刚洒进来",
    "afternoon": "午后的阳光暖洋洋的",
    "evening": "天色渐渐暗了下来",
    "night": "夜深了，四周很安静",
}


@dataclass
class LLMConfig:
    """LLM集成完整配置"""

    # ── 模型 ──
    model_path: str = MODEL_PATH
    model_dir: str = MODEL_DIR
    model_filename: str = MODEL_FILENAME
    model_family: str = "deepseek-r1-distill-qwen"
    model_size: str = "1.5B"
    quantization: str = "Q4_0"
    estimated_memory_gb: float = 1.0

    # ── 推理参数 ──
    n_ctx: int = LLAMA_CPP_N_CTX
    n_threads: int = LLAMA_CPP_N_THREADS
    n_batch: int = LLAMA_CPP_N_BATCH
    max_tokens: int = LLAMA_CPP_MAX_TOKENS
    temperature: float = LLAMA_CPP_TEMPERATURE
    top_p: float = LLAMA_CPP_TOP_P
    top_k: int = LLAMA_CPP_TOP_K
    repeat_penalty: float = LLAMA_CPP_REPEAT_PENALTY
    stop_sequences: List[str] = field(default_factory=lambda: LLAMA_CPP_STOP)

    # ── 服务器模式 ──
    use_server: bool = False
    server_host: str = LLAMA_SERVER_HOST
    server_port: int = LLAMA_SERVER_PORT
    server_ctx_size: int = LLAMA_SERVER_CTX_SIZE

    # ── 缓存 ──
    cache_ttl_seconds: float = CACHE_TTL_SECONDS
    cache_max_size: int = CACHE_MAX_SIZE
    emotion_bucket_size: int = EMOTION_BUCKET_SIZE

    # ── 降级 ──
    llm_timeout_ms: float = LLM_TIMEOUT_MS
    fallback_cooldown_seconds: float = FALLBACK_COOLDOWN_SECONDS
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES
    health_check_interval: float = LLM_HEALTH_CHECK_INTERVAL

    # ── 并发 ──
    max_concurrent: int = MAX_CONCURRENT_GENERATIONS
    queue_size: int = GENERATION_QUEUE_SIZE

    # ── 提示词 ──
    system_prompt_template: str = SYSTEM_PROMPT_TEMPLATE
    user_prompt_template: str = USER_PROMPT_TEMPLATE
