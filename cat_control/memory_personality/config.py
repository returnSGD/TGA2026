"""
《猫语心声》 —— 记忆系统与性格过滤器配置

严格遵循 技术策划案v2 §4.3 & §4.4
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 重用 rl_environment 的常量定义
from rl_environment.config import (
    INTENT_LIST, PERSONALITY_KEYS, PERSONALITY_DIM,
    EMOTION_DIM, ENV_FEATURE_DIM, MEMORY_EMBED_DIM, TOP_K_MEMORIES,
    STATE_DIM, PLAYER_ACTION_DIM,
    WORK_MEMORY_SIZE, LONG_MEMORY_CAP, MEMORY_IMPORTANCE_THRESHOLD,
    INTENT_PERSONALITY_MATRIX, PERSONALITY_BEHAVIOR_PARAMS,
    PERSONALITY_FORBIDDEN_WORDS, EVENT_IMPORTANCE_MID,
    CAT_CONFIGS,
)


@dataclass
class MemoryConfig:
    """记忆系统配置"""

    # ══ 记忆容量 ══
    work_memory_size: int = WORK_MEMORY_SIZE        # 工作记忆环形队列 20条
    long_memory_cap: int = LONG_MEMORY_CAP          # 长期记忆优先级队列 500条
    importance_threshold: float = MEMORY_IMPORTANCE_THRESHOLD  # 进入长期记忆的阈值 4.0

    # ══ 向量嵌入 ══
    embed_dim: int = MEMORY_EMBED_DIM               # 嵌入向量维度 128
    embed_model_name: str = "all-MiniLM-L6-v2"      # Sentence Transformer 模型
    embed_device: str = "cpu"                       # 嵌入模型设备
    embed_batch_size: int = 32                      # 批量嵌入批次大小
    top_k_retrieval: int = TOP_K_MEMORIES           # 检索Top-K 3条

    # ══ 向量数据库 ══
    vector_db_backend: str = "numpy"                # "sqlite_vec" | "chroma" | "numpy"
    vector_db_path: str = os.path.join(
        BASE_DIR, "memory_personality", "data", "memory_vectors.db"
    )

    # ══ 记忆时间衰减 ══
    time_decay_max_ttl: float = 90.0 * 144          # 最大TTL (游戏内天数 × 每天tick)
    time_decay_min: float = 0.1                     # 最小衰减系数

    # ══ 记忆压缩 ══
    compress_age_days: float = 90.0                 # 超过此天数的记忆触发压缩
    compress_min_count: int = 10                    # 压缩触发的最小旧记忆数量

    # ══ RL集成 ══
    query_emotion_weight: float = 0.6               # 查询向量中情绪的权重
    query_env_weight: float = 0.4                   # 查询向量中环境的权重

    # ══ 性格过滤器 ══
    personality_dim: int = PERSONALITY_DIM
    intent_num: int = len(INTENT_LIST)
    intent_trait_matrix: Dict = field(default_factory=lambda: INTENT_PERSONALITY_MATRIX)
    behavior_param_table: Dict = field(default_factory=lambda: PERSONALITY_BEHAVIOR_PARAMS)
    forbidden_words: Dict = field(default_factory=lambda: PERSONALITY_FORBIDDEN_WORDS)
    trait_threshold: float = 0.7                    # 性格维度超过此值触发禁用词

    # ══ 路径 ══
    data_dir: str = os.path.join(BASE_DIR, "memory_personality", "data")
    export_dir: str = os.path.join(BASE_DIR, "memory_personality", "exports")

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)


# 猫咪性格定义（与 rl_environment.config.CAT_CONFIGS 对应）
CAT_PERSONALITIES = {
    cat_id: cfg["personality"]
    for cat_id, cfg in CAT_CONFIGS.items()
}

# 记忆查询向量的构建维度
QUERY_VECTOR_DIM = EMOTION_DIM + ENV_FEATURE_DIM  # 5 + 5 = 10维
