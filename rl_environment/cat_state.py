"""
《猫语心声》 —— 猫咪状态与记忆系统

CatState: MDP状态向量的完整定义
MemoryItem: 记忆单元
MemoryManager: 两级记忆系统（工作记忆 + 长期记忆）
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import heapq
import numpy as np

from .config import (
    EMOTION_DIM, PHYSICAL_DIM, ENV_FEATURE_DIM, RELATION_DIM,
    PERSONALITY_DIM, PERSONALITY_KEYS, MEMORY_EMBED_DIM,
    WORK_MEMORY_SIZE, LONG_MEMORY_CAP, MEMORY_IMPORTANCE_THRESHOLD,
    TOP_K_MEMORIES, NEED_MIN, NEED_MAX, TRUST_MIN, TRUST_MAX,
)


@dataclass
class CatState:
    """猫咪当前状态（MDP状态）"""

    # 性格嵌入（固定，不会改变）
    personality_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(PERSONALITY_DIM, dtype=np.float32))

    # 情绪向量 (5维)：饥饿, 恐惧, 好奇, 舒适, 社交需求
    # 值域 [0, 1]，越高表示该情绪越强烈
    emotion_vector: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.3, 0.5, 0.5, 0.3], dtype=np.float32))

    # 生理向量 (3维)：精力, 健康, 体温
    # 精力[0,1] 低=疲劳, 健康[0,1] 低=生病, 体温[0,1] 0.5=正常
    physical_vector: np.ndarray = field(
        default_factory=lambda: np.ones(PHYSICAL_DIM, dtype=np.float32))

    # 信任度 (0-100)
    trust_level: float = 20.0

    # 压力值 (0-100)
    stress_level: float = 50.0

    # 环境特征 (5维)：区域舒适度, 刺激度, 卫生度, 光照, 噪音
    environment_features: np.ndarray = field(
        default_factory=lambda: np.zeros(ENV_FEATURE_DIM, dtype=np.float32))

    # 关系向量 (4维)：对玩家好感, 最近猫咪亲密度均值, 最近猫咪敌意度均值, 社交排名
    relation_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(RELATION_DIM, dtype=np.float32))

    # 空间状态
    current_room_id: int = 0
    position: Tuple[int, int] = (0, 0)

    # 当前行为状态
    current_intent: str = "idle_wander"
    current_action: str = "none"
    action_progress: float = 0.0  # 0~1，当前动作完成进度
    action_target: Optional[Tuple[int, int]] = None

    # 社交状态
    nearby_cats: List[str] = field(default_factory=list)
    recent_interactions: List[str] = field(default_factory=list)

    # 状态快照
    last_intent_change_tick: int = 0
    consecutive_intent_failures: int = 0
    is_sleeping: bool = False
    is_eating: bool = False
    is_hiding: bool = False

    def clone(self) -> CatState:
        """深拷贝状态"""
        import copy
        new_state = CatState()
        new_state.personality_vector = self.personality_vector.copy()
        new_state.emotion_vector = self.emotion_vector.copy()
        new_state.physical_vector = self.physical_vector.copy()
        new_state.trust_level = self.trust_level
        new_state.stress_level = self.stress_level
        new_state.environment_features = self.environment_features.copy()
        new_state.relation_vector = self.relation_vector.copy()
        new_state.current_room_id = self.current_room_id
        new_state.position = self.position
        new_state.current_intent = self.current_intent
        new_state.current_action = self.current_action
        new_state.action_progress = self.action_progress
        new_state.action_target = self.action_target
        new_state.nearby_cats = list(self.nearby_cats)
        new_state.recent_interactions = list(self.recent_interactions)
        new_state.last_intent_change_tick = self.last_intent_change_tick
        new_state.consecutive_intent_failures = self.consecutive_intent_failures
        new_state.is_sleeping = self.is_sleeping
        new_state.is_eating = self.is_eating
        new_state.is_hiding = self.is_hiding
        return new_state

    def to_state_vector(self) -> np.ndarray:
        """将状态各部分拼接为原始向量（不含玩家行为和记忆）"""
        return np.concatenate([
            self.personality_vector,
            self.emotion_vector,
            self.physical_vector,
            [self.trust_level / 100.0],
            self.environment_features,
            self.relation_vector,
        ]).astype(np.float32)

    @property
    def hunger(self) -> float:
        return self.emotion_vector[0]

    @hunger.setter
    def hunger(self, v: float):
        self.emotion_vector[0] = np.clip(v, NEED_MIN / 100.0, NEED_MAX / 100.0)

    @property
    def fear(self) -> float:
        return self.emotion_vector[1]

    @fear.setter
    def fear(self, v: float):
        self.emotion_vector[1] = np.clip(v, NEED_MIN / 100.0, NEED_MAX / 100.0)

    @property
    def curiosity(self) -> float:
        return self.emotion_vector[2]

    @curiosity.setter
    def curiosity(self, v: float):
        self.emotion_vector[2] = np.clip(v, NEED_MIN / 100.0, NEED_MAX / 100.0)

    @property
    def comfort(self) -> float:
        return self.emotion_vector[3]

    @comfort.setter
    def comfort(self, v: float):
        self.emotion_vector[3] = np.clip(v, NEED_MIN / 100.0, NEED_MAX / 100.0)

    @property
    def social_need(self) -> float:
        return self.emotion_vector[4]

    @social_need.setter
    def social_need(self, v: float):
        self.emotion_vector[4] = np.clip(v, NEED_MIN / 100.0, NEED_MAX / 100.0)

    @property
    def energy(self) -> float:
        return self.physical_vector[0]

    @energy.setter
    def energy(self, v: float):
        self.physical_vector[0] = np.clip(v, 0.0, 1.0)

    @property
    def health(self) -> float:
        return self.physical_vector[1]

    @health.setter
    def health(self, v: float):
        self.physical_vector[1] = np.clip(v, 0.0, 1.0)

    def summary(self) -> str:
        """生成可读的状态摘要"""
        trait_str = ", ".join(
            f"{k}={v:.1f}" for k, v in zip(PERSONALITY_KEYS, self.personality_vector) if v > 0.3
        )
        return (
            f"饥饿:{self.hunger:.0%} 恐惧:{self.fear:.0%} 好奇:{self.curiosity:.0%} "
            f"舒适:{self.comfort:.0%} 社交:{self.social_need:.0%} | "
            f"精力:{self.energy:.0%} 健康:{self.health:.0%} | "
            f"信任:{self.trust_level:.0f} 压力:{self.stress_level:.0f} | "
            f"意图:{self.current_intent} | "
            f"性格:[{trait_str}]\n           "
            f"位置:{self.position} 房间:{self.current_room_id}"
        )


@dataclass
class MemoryItem:
    """记忆单元（可排序，按importance降序）"""
    importance: float
    desc: str = ""
    timestamp: float = 0.0
    embedding: Optional[np.ndarray] = None
    event_type: str = "daily"

    def __lt__(self, other: 'MemoryItem') -> bool:
        return self.importance > other.importance


class MemoryManager:
    """
    两级记忆系统：
    - 工作记忆：固定大小环形队列（20条）
    - 长期记忆：优先级队列（500条上限），基于重要性排序
    """

    def __init__(self, embed_dim: int = MEMORY_EMBED_DIM):
        self.working_memory: deque[MemoryItem] = deque(maxlen=WORK_MEMORY_SIZE)
        self.long_term_memory: List[MemoryItem] = []
        self.embed_dim = embed_dim
        self.total_memories_stored = 0

    def add_memory(self, item: MemoryItem):
        """添加记忆到工作记忆，重要性超阈值则进入长期记忆"""
        self.working_memory.append(item)
        self.total_memories_stored += 1
        if item.importance >= MEMORY_IMPORTANCE_THRESHOLD:
            heapq.heappush(self.long_term_memory, item)
            while len(self.long_term_memory) > LONG_MEMORY_CAP:
                heapq.heappop(self.long_term_memory)

    def get_recent_memories(self, k: int = 5) -> List[MemoryItem]:
        """获取最近k条工作记忆"""
        return list(self.working_memory)[-k:]

    def query_similar(self, query_vector: np.ndarray,
                      top_k: int = TOP_K_MEMORIES) -> List[MemoryItem]:
        """
        基于向量余弦相似度检索最相关的长期记忆。
        当记忆库为空或嵌入未生成时，返回空列表。
        """
        if not self.long_term_memory:
            return []

        scored = []
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        for mem in self.long_term_memory:
            if mem.embedding is not None:
                emb_norm = mem.embedding / (np.linalg.norm(mem.embedding) + 1e-8)
                sim = float(np.dot(query_norm, emb_norm))
                scored.append((sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def get_memory_embeddings(self, query_vector: np.ndarray,
                              top_k: int = TOP_K_MEMORIES) -> List[np.ndarray]:
        """检索Top-K记忆嵌入，不足则零向量填充"""
        memories = self.query_similar(query_vector, top_k)
        result = []
        for i in range(top_k):
            if i < len(memories) and memories[i].embedding is not None:
                result.append(memories[i].embedding)
            else:
                result.append(np.zeros(self.embed_dim, dtype=np.float32))
        return result

    def compress_old_memories(self, current_time: float,
                              days_threshold: float = 90.0):
        """压缩超过阈值的旧记忆为摘要向量"""
        old_mems = []
        kept_mems = []
        time_threshold = current_time - days_threshold * 144  # 每天144 tick

        for mem in self.long_term_memory:
            if mem.timestamp < time_threshold:
                old_mems.append(mem)
            else:
                kept_mems.append(mem)

        if old_mems:
            # 生成摘要：取均值嵌入
            embeddings = [m.embedding for m in old_mems if m.embedding is not None]
            if embeddings:
                summary_embed = np.mean(embeddings, axis=0)
            else:
                summary_embed = np.zeros(self.embed_dim, dtype=np.float32)

            descs = [m.desc for m in old_mems[:5]]
            summary_item = MemoryItem(
                desc=f"[压缩记忆] {'; '.join(descs)}...",
                timestamp=current_time,
                importance=min(m.importance for m in old_mems) if old_mems else 3.0,
                embedding=summary_embed,
                event_type="summary",
            )

            self.long_term_memory = kept_mems
            heapq.heappush(self.long_term_memory, summary_item)

    def size_working(self) -> int:
        return len(self.working_memory)

    def size_long_term(self) -> int:
        return len(self.long_term_memory)

    def summary(self) -> str:
        return (f"记忆系统: 工作记忆 {self.size_working()}/{WORK_MEMORY_SIZE}, "
                f"长期记忆 {self.size_long_term()}/{LONG_MEMORY_CAP}, "
                f"总计存储 {self.total_memories_stored} 条")
